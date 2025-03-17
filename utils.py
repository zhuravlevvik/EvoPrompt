import re
import yaml
import random
import string
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import torch
import numpy as np
from src.data.classification import *
from src.data.generation import *
from src.data.multi_task import *
from src.data.qa import *


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def remove_punctuation(s):
    translator = str.maketrans('', '', string.punctuation)
    return s.translate(translator)


def first_appear_pred(text, verbalizer_dict, logger):
    text = text.lower()
    verbalizer_dict = [k.lower() for k in verbalizer_dict]
    for word in text.split():
        if word in verbalizer_dict:
            return word
    # logger.info("cannot decode {}".format(text))
    return ""


def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def read_lines(file_, sample_indices=None):
    ret = []
    if sample_indices:
        sample_indices.sort()
        with open(file_, 'r') as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    ret.append(line.rstrip())
        return ret
    else:
        with open(file_, 'r') as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines]


def json2list(file):
    with open(file, 'r') as f:
        lines = json.load(f)
    return lines


def extract_numbers(string):
    return [int(num) for num in re.findall(r'\d+', string)][0]


def extract_n_samples_per_class(src, tgt, n, dataset):
    src_new = []
    tgt_new = []
    for label in set(tgt):
        cur_src = [src[i] for i, value in enumerate(tgt) if value == label]
        cur_tgt = [tgt[i] for i, value in enumerate(tgt) if value == label]
        rand_indices = random.sample(range(len(cur_src)), n)
        # print(rand_indices)
        src_new += [cur_src[i] for i in rand_indices]
        tgt_new += [cur_tgt[i] for i in rand_indices]
    tgt_new = [e[1:] for e in tgt_new] if dataset != 'agnews' else tgt_new
    return src_new, tgt_new


def batchify(data, batch_size=16):
    batched_data = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i + batch_size])
    return batched_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_log(log_path, log_name="basic"):
    print("Setting up log for", log_name)
    logger = logging.getLogger(log_name)
    if not logger.handlers:
        # log_path = os.path.join("logs", log_name)
        logger.setLevel(logging.DEBUG)
        file_handler = TimedRotatingFileHandler(
            filename=log_path, when="MIDNIGHT", interval=1, backupCount=30
        )
        file_handler.suffix = "%Y-%m-%d.log"
        file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
        stream_handler = logging.StreamHandler()
        # formatter = logging.Formatter("[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s")
        formatter = logging.Formatter("[%(asctime)s] - %(message)s")

        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    return logger


def k_init_pop(initial_mode, init_population, k):
    if initial_mode == "topk":
        population = [i for i in init_population[:k]]
    elif initial_mode == "para_topk":
        population = [i for i in init_population[: k // 2]]
    elif initial_mode == "para_bottomk":
        population = [i for i in init_population[-k // 2 :]]
    elif initial_mode == "para_randomk":
        population = random.sample(init_population, k // 2)
    elif initial_mode == "randomk":
        population = random.sample(init_population, k)
    elif initial_mode == "bottomk":
        population = [i for i in init_population[-k:]]
    return population


def cal_mean_std(results):
    if results[0] < 1.0:
        results = [result * 100 for result in results]
    mean = np.mean(results)
    std = np.std(results)
    return round(mean, 2), round(std, 2)


def load_dataset(
    name: str,
    tokenizer=None,
    prompt: str = "",
):
    ds = None
    match name:
        case "sst-2":
            ds = SST2Dataset(
                tokenizer=tokenizer,
                data_path="~/CoolPrompt/data/sst-2/test-00000-of-00001.parquet",
                config_path="~/CoolPrompt/data",
                prompt=prompt,
            )
        case _:
            raise ValueError(f"Invalid dataset name: {name}")

    return ds


if __name__ == '__main__':
    dev_src, dev_tgt, test_src, test_tgt = load_sum_data('sam', 5, 100)
    lengths = [len(i) for i in dev_src]
    from collections import Counter
    print(dict(Counter(lengths))[0])