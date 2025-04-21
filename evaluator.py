import os
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import setup_log, batchify
from src.evaluation.evaluator import (
    TextClassificationEvaluator,
    GenerationEvaluator
)
from src.utils.load_dataset import load_dataset


class Evaluator(object):
    def __init__(self, args) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dataset = args.dataset

        self.model = LLM(
            model=args.language_model,
            dtype=torch.float16,
            trust_remote_code=True,
            # tensor_parallel_size=1,
            # pipeline_parallel_size=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.language_model, padding_side="left", use_fast=False
        )

        self.model_generate_args = {
            "max_tokens": 50,
            "stop_token_ids": [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
            ]
        }

        self.public_out_path = args.output
        if not os.path.exists(self.public_out_path):
            os.makedirs(self.public_out_path)
        self.logger = setup_log(
            os.path.join(self.public_out_path, "evol.log")
        )
        logger = self.logger
        logger.info("=" * 50)
        logger.info(f"dev data: {args.dev_file}")
        logger.info(f"test data: {args.test_file}")
        self.args = args
        logger.info(
            "\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items())
        )
        logger.info("=" * 50)

    def forward(self, prompt: str = "", test=False):
        sample = self.args.sample_num
        split = 'test' if test else 'train'
        dataset = load_dataset(
            self.dataset,
            split=split,
            tokenizer=self.tokenizer,
            prompt=prompt,
            sample=sample
        )
        scores = self.metrics_evaluator.evaluate_vllm(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_ds=dataset,
            batch_size=self.args.batch_size,
            model_generate_args=self.model_generate_args
        )
        return [scores[self.args.metric]]

    def llm_query(self, data, **config):
        sampling_params = {
            "max_tokens": 1000,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        sampling_params.update(**config)
        sampling_params = SamplingParams(**sampling_params)

        if not isinstance(data, list):
            data = [data]

        data = batchify(data)
        results = []

        for batch in tqdm(data):
            answers = self.model.generate(
                prompts=batch, sampling_params=sampling_params, use_tqdm=False
            )

            outputs = [answer.outputs[0].text for answer in answers]
            results.extend(outputs)

        if len(results) == 1:
            results = results[0]
        return results

    def paraphrase(self, sentence, **kwargs):
        if isinstance(sentence, list):
            resample_template = [
                f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
                for s in sentence
            ]

        else:
            resample_template = [f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"]

        return self.llm_query(resample_template, **kwargs)


class CLSEvaluator(Evaluator):
    def __init__(self, args):
        super(CLSEvaluator, self).__init__(args)
        self.metrics_evaluator = TextClassificationEvaluator()


class GENEvalutator(Evaluator):
    def __init__(self, args):
        super(GENEvalutator, self).__init__(args)
        self.metrics_evaluator = GenerationEvaluator()
