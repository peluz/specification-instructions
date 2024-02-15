import openai
from tqdm.auto import tqdm
import random
import time
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download
import torch
import numpy as np



def openai_request(prompts, api_token, format_instruction,  temperature=0.0, max_tokens=20):
    openai.api_key = api_token
    responses = []
    requests = [[{"role": "user", "content": prompt + f"{format_instruction}"}] for prompt in prompts]
    for request in tqdm(requests):
        while True:
            num_retries = 0
            delay = 1.
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=request,
                max_tokens=max_tokens,
                temperature=0,
                )
                responses.append(response)
                break
            except Exception as e:
                num_retries += 1
                print(e)
 
                # Check if max retries has been reached
                if num_retries > 10:
                    raise Exception(
                        f"Maximum number of retries (10) exceeded."
                    )
 
                # Increment the delay
                delay *= 2 * (1 + random.random())
                print(f"Retrying with delay {delay}")
 
                # Sleep for the delay
                time.sleep(delay)
    all_preds = [response["choices"][0]["message"]["content"] for response in responses]
    return responses, all_preds

def load_model(model_url, num_gpus=1):
    if num_gpus > 8:
        device_map  = "balanced_low_0"
    else:
        device_map = "auto"
    if num_gpus > 1:
        with init_empty_weights():
            config = AutoConfig.from_pretrained(model_url)
            model = AutoModelForSeq2SeqLM.from_config(config)
        weights_location = snapshot_download(repo_id=model_url, allow_patterns=["pytorch_model*"])
        model = load_checkpoint_and_dispatch(
            model, weights_location, device_map=device_map, no_split_module_classes=["T5Block"]
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_url).to(0)
    return model

def get_max_batch_size(inputs, pipe, model_url, out_file, max_new_tokens):
    if out_file.exists():
        with open(out_file, "r") as file:
            bs = file.readline().strip()
        if len(bs)>0:
            print(f"Max batch size is {bs}")  
            return int(bs)
    if "zephyr" in model_url.lower():
        inputs_tokenized = [pipe.tokenizer(x)["input_ids"] for x in inputs]
    else:
        inputs_tokenized = [pipe.tokenizer(x)["input_ids"] for x in inputs]
    biggest = np.argmax([len(x) for x in inputs_tokenized])
    biggest_prompt = inputs[biggest]
    print(f"Number of examples: {len(inputs_tokenized)}")
    print(f"Size of biggest sample: {len(inputs_tokenized[biggest])} tokens")
    print(biggest_prompt)

    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        print(f"Trying with bs {bs}\n\n")
        prompts = bs*[biggest_prompt]
        dataset = Dataset.from_list([{"prompt": prompt} for prompt in prompts])
        try:
            for out in tqdm(pipe(KeyDataset(dataset, "prompt"), max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, top_p=1.0, batch_size=bs)):
                print(out)
        except Exception as e:
            print(e)
            bs = bs//2
            break
    print(f"Final batch size: {bs}")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as file:
        file.write(str(bs))
    return bs

def get_preds(prompts, pipe, model_url, format_instruction, out_file, max_new_tokens=20):
    if "zephyr" in model_url.lower():
        inputs = []
        for prompt in prompts:
            input = [
                        {"role": "user", "content": prompt + f"{format_instruction}"},
                    ]
            inputs.append(input)
        inputs = [pipe.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in inputs]
    else:
        pipe.tokenizer.model_max_length = 4096
        inputs = prompts
    bs = get_max_batch_size(inputs, pipe, model_url, out_file, max_new_tokens)
    dataset = Dataset.from_list([{"prompt": prompt} for prompt in inputs])
    if "zephyr" in model_url.lower():
        try:
            all_preds = []
            for out in tqdm(pipe(KeyDataset(dataset, "prompt"), max_new_tokens=max_new_tokens, do_sample=False, temperature=0, top_p=1.0, return_full_text=False, batch_size=bs), total=len(dataset)):
                all_preds.append(out)
        except torch.cuda.OutOfMemoryError:
            all_preds = []
            bs /= 2
            for out in tqdm(pipe(KeyDataset(dataset, "prompt"), max_new_tokens=max_new_tokens, do_sample=False, temperature=0, top_p=1.0, return_full_text=False, batch_size=bs), total=len(dataset)):
                all_preds.append(out)
    else:
        try:
            all_preds = []
            for out in tqdm(pipe(KeyDataset(dataset, "prompt"), max_new_tokens=max_new_tokens, do_sample=False, temperature=0, top_p=1.0, batch_size=bs), total=len(dataset)):
                all_preds.append(out)
        except torch.cuda.OutOfMemoryError:
            all_preds = []
            bs /= 2
            for out in tqdm(pipe(KeyDataset(dataset, "prompt"), max_new_tokens=max_new_tokens, do_sample=False, temperature=0, top_p=1.0, batch_size=bs), total=len(dataset)):
                all_preds.append(out)
    return all_preds