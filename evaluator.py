import json
import os
from tqdm import tqdm
import numpy as np
import random
import sys
import time
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from datasets import Dataset as Dataset2
from sacrebleu.metrics import BLEU, CHRF, TER

from utils import *
from dataset import TextDataset
from llm_client import *
from metrics import *
from src.evaluation.evaluator import TextClassificationEvaluator, GenerationEvaluator


class Evaluator(object):
    def __init__(self, args) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = args.dataset

        self.model = AutoModelForCausalLM.from_pretrained(
            args.language_model, torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.language_model, padding_side="left", use_fast=False
        )

        self.model_generate_args = {
            max_new_tokens: 50,
            eos_token_id: [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
            ]
        }

        self.public_out_path = args.output
        if not os.path.exists(self.public_out_path):
            os.makedirs(self.public_out_path)
        self.logger = setup_log(os.path.join(self.public_out_path, f"evol.log"))
        logger = self.logger
        logger.info("=" * 50)
        logger.info(f"dev data: {args.dev_file}")
        logger.info(f"test data: {args.test_file}")
        self.args = args
        logger.info("\n\t" + "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
        logger.info("=" * 50)

    def forward(self, prompt: str = ""):
        dataset = load_dataset(
            self.dataset,
            tokenizer=self.tokenizer,
            prompt=prompt,
        )
        scores = self.metrics_evaluator.evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_ds=dataset,
            batch_size=self.args.batch_size,
            model_generate_args=self.model_generate_args
        )
        return scores


class CLSEvaluator(Evaluator):
    def __init__(self, args):
        super(CLSEvaluator, self).__init__(args)
        self.metrics_evaluator = TextClassificationEvaluator()


class SumEvaluator(Evaluator):
    def __init__(self, args):
        super(SumEvaluator, self).__init__(args)
        self.metrics_evaluator = GenerationEvaluator()


class SimEvaluator(Evaluator):
    def __init__(self, args):
        super(SimEvaluator, self).__init__(args)
        self.metrics_evaluator = GenerationEvaluator()
