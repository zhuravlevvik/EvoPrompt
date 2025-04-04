import sys
import os
# This code enables using of "src.data" imports in vs code (when you're launching it directly from notebooks directory)
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)

from transformers import AutoTokenizer
import torch

print(torch.cuda.is_available())

torch.manual_seed(42)

from vllm import LLM

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = LLM(model=model_name, dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", padding_side='left')
from vllm import SamplingParams


params = {
    "temperature": 0.6,
    "max_tokens": 2500,
    "stop_token_ids": [tokenizer.eos_token_id],
}
sampling_params = SamplingParams(**params)


import pandas as pd
import os


df = pd.read_parquet(os.path.expanduser('~/autoprompting_data/gsm8k/train-00000-of-00001.parquet'))


def make_examples(num_samples):
    sampled = df.sample(num_samples)

    e = [f"Input: \"{row['input']}\"   Output: \"{row['target']}\""  for _, row in sampled.iterrows()]
    e = '\n'.join(e)
    return e


def ape(num_samples, num_samples_str, num_prompts):
    cnt = 0
    while True:
        e = make_examples(num_samples)

        inputs = [
            f"""I gave a friend an instruction and {num_samples_str} inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:

        {e}

        Write the instuction I gave to my friend and bracket it with <prompt></prompt>
        <think>
        """
        ]

        answers = model.generate(
            prompts=inputs, sampling_params=sampling_params, use_tqdm=False
        )

        outputs = [answer.outputs[0].text for answer in answers]
        text = outputs[0]
        text = text[text.find('</think>'):]
        pos1 = text.find('<prompt>')
        pos2 = text.find('</prompt>')
        if pos1 == -1 or pos2 == -1:
            continue
        else:
            print(text[pos1 + 8:pos2])
            print('-' * 50)
            cnt += 1
            if cnt == num_prompts:
                break


def main():
    ape(
        num_samples=5,
        num_samples_str="five",
        num_prompts=15
    )
    
    
if __name__ == "__main__":
    main()
 