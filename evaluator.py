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
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from coolprompt.evaluator import Evaluator as CoolEval, validate_and_create_metric
from coolprompt.utils.var_validation import validate_task


class Evaluator(object):
    def __init__(self, args, model, train_dataset, train_targets, valid_dataset, valid_targets) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_targets = train_targets
        self.valid_targets = valid_targets

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
        if test:
            dataset = self.valid_dataset
            target = self.valid_targets
        else:
            dataset = self.train_dataset
            target = self.train_targets
        score = self._evaluator.evaluate(prompt, dataset, target)
        return [score]

    def llm_query(self, data, **config):
        if not isinstance(data, list):
            data = [data]

        answers = self.model.batch(data)

        answers = [a.content
                   if isinstance(a, AIMessage)
                   else a for a in answers]

        return answers

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
    def __init__(self, args, model, train_dataset, train_targets, valid_dataset, valid_targets):
        super(CLSEvaluator, self).__init__(args, model, train_dataset, train_targets, valid_dataset, valid_targets)
        task = validate_task("classification")
        metric = validate_and_create_metric(task, args.metric)
        self._evaluator = CoolEval(self.model, task, metric)


class GENEvalutator(Evaluator):
    def __init__(self, args, model, train_dataset, train_targets, valid_dataset, valid_targets):
        super(GENEvalutator, self).__init__(args, model, train_dataset, train_targets, valid_dataset, valid_targets)
        task = validate_task("generation")
        metric = validate_and_create_metric(task, args.metric)
        self._evaluator = CoolEval(self.model, task, metric)
