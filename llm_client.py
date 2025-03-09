from tqdm import tqdm
import time
from utils import remove_punctuation, batchify

def llm_query(data, client, type, task, **config):
    hypos = []
    response = None
    model_name = "davinci" if "davinci" in type else "turbo"
    # batch
    if isinstance(data, list):
        batch_data = batchify(data, 20)
        for batch in tqdm(batch_data):
            retried = 0
            request_data = form_request(batch, model_name, **config)
            if "davinci" in type:
                # print(request_data)
                while True:
                    try:
                        response = openai.Completion.create(**request_data)
                        response = response["choices"]
                        response = [r["text"] for r in response]
                        break
                    except Exception as e:
                        error = str(e)
                        print("retring...", error)
                        second = extract_seconds(error, retried)
                        retried = retried + 1
                        time.sleep(second)
            else:
                response = []
                for data in tqdm(batch):
                    request_data = form_request(data, type, **config)
                    while True:
                        try:
                            result = openai.ChatCompletion.create(**request_data)
                            result = result["choices"][0]["message"]["content"]
                            # print(result)
                            response.append(result)
                            break
                        except Exception as e:
                            error = str(e)
                            print("retring...", error)
                            second = extract_seconds(error, retried)
                            retried = retried + 1
                            time.sleep(second)

            # print(response)
            if task:
                results = [str(r).strip().split("\n\n")[0] for r in response]
            else:
                results = [str(r).strip() for r in response]
            # print(results)
            # results = [str(r['text']).strip() for r in response]
            # print(results)
            hypos.extend(results)
    else:
        retried = 0
        while True:
            try:
                print(type)
                result = ""
                if "turbo" in type or 'gpt4' in type:
                    request_data = form_request(data, type, **config)
                    response = openai.ChatCompletion.create(**request_data)
                    result = response["choices"][0]["message"]["content"]
                    break
                else:
                    request_data = form_request(data, type=type, **config)
                    response = openai.Completion.create(**request_data)["choices"][ 0 ]["text"]
                    result = response.strip()
            except Exception as e:
                error = str(e)
                print("retring...", error)
                second = extract_seconds(error, retried)
                retried = retried + 1
                time.sleep(second)
        if task:
            result = result.split("\n\n")[0]

        hypos = result
    return hypos


def paraphrase(sentence, client, type, **kwargs):
    if isinstance(sentence, list):
        resample_template = [
            f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
            for s in sentence
        ]

    else:
        resample_template = f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"
    # print(resample_template)
    results = llm_query(resample_template, client, type, False, **kwargs)
    return results


def llm_cls(dataset, client=None, type=None, **config):
    hypos = []
    results = llm_query(dataset, client=client, type=type, task=True, **config)
    if isinstance(results, str):
        results = [results]
    hypos = [remove_punctuation(r.lower()) for r in results]

    return hypos



if __name__ == "__main__":
    llm_client = None
    llm_type = 'turbo'
    start = time.time()
    data = ["""Q: Tom bought a skateboard for $ 9.46 , and spent $ 9.56 on marbles . Tom 
also spent $ 14.50 on shorts . In total , how much did Tom spend on toys ?                                                 
A: Let's think step by step. """]
    config = llm_init(auth_file="auth.yaml", llm_type=llm_type, setting="default")
    para = llm_query(
        data[0], client=llm_client, type=llm_type, task=False, temperature=0, **config
    )
    print(para)
    end = time.time()
    print(end - start)
