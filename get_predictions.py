from data_sets.data_utils import get_suite, load_hsd_dataset
from models.inference import *
from datasets import load_dataset, concatenate_datasets
from utils.util import initialize_seeds
from utils.prompts import *
from transformers import DataCollatorWithPadding, T5TokenizerFast
import config
import json
from pathlib import Path
import numpy as np
import pickle
import argparse
import torch


def main(task="sa",model_url=None, bs=None, chatGPT=False, methods="all", ask_rule=False, add_examples=False, from_chatGPT=False):
    initialize_seeds()
    if not chatGPT:
        model_name = model_url.split("/")[-1]
    else:
        model_name = "chatGPT"
    if task == "sa":
        dataset_name = "sst2"
        suite_path = config.sa_path
    elif task == "pi":
        dataset_name = "qqp"
        suite_path = config.pi_path
    elif task == "rc":
        dataset_name = "squad"
        suite_path = config.rc_path
    if task in ["sa", "pi"]:
        dataset = load_dataset("glue", dataset_name)
    elif task == "hsd":
        davidson = load_hsd_dataset("davidson2017")
        founta = load_hsd_dataset("founta2018")
    else:
        dataset = load_dataset(dataset_name)
    if task != "hsd":
        dataset_train = dataset["train"]
        dataset_test = dataset["validation"]
    else:
        davidson_train, davidson_test = davidson["train"], davidson["test"]
        founta_train, founta_test = founta["train"], founta["test"]
    if task != "hsd":
        suite_test = get_suite(suite_path)["test"]
    else:
        suite_test = get_suite(config.hatecheck_path, hateCheck=True).rename_column("label_gold", "label").rename_column("test_case", "text")["test"]

    if add_examples:
        if task == "hsd":
            davidson_examples_per_label = [davidson_train.filter(lambda x: x["label"] == label) for label in [0, 1]]
            founta_examples_per_label = [founta_train.filter(lambda x: x["label"] == label) for label in [0, 1]]
            dataset_train = concatenate_datasets([davidson_train, founta_train])
            examples_per_label = [dataset_train.filter(lambda x: x["label"] == label) for label in [0, 1]]
        elif task == "rc":
            examples_per_label = dataset_train
        else:
            examples_per_label = [dataset_train.filter(lambda x: x["label"] == label) for label in [0, 1]]
    else:
        davidson_examples_per_label = founta_examples_per_label = examples_per_label = None

    if not chatGPT:
        num_gpus = torch.cuda.device_count()
        print(num_gpus)
        tokenizer = T5TokenizerFast.from_pretrained(model_url)
        model = load_model(model_url, num_gpus)
        if num_gpus > 1:
            print(model.hf_device_map)
        data_collator = DataCollatorWithPadding(tokenizer)
    else:
        api_token = config.openai_key
        if not add_examples:
            if task != "rc":
                output_format = "Output exactly one of the options.\n"
            else:
                output_format = "Output a concise, minimal answer.\n"
        else:
            output_format = ""
    max_new_tokens=90 if task=="rc" else 20
    suffix = ""
    if from_chatGPT:
        suffix += "_from_chatGPT"
    if add_examples:
        suffix += "_example"
    if ask_rule:
        max_new_tokens += 150
        suffix += "_with_rules"


    with open(f"./data/{task}/suite/class_to_funcs.pkl", "rb") as file:
        class_to_funcs = pickle.load(file)
    if not from_chatGPT:
        with open(f"./data/{task}/suite/func_desc.pkl", "rb") as file:
            func_desc = pickle.load(file)
    else:
        with open(f"./data/chatGPTgeneratedRules/{task}/rules.json", "r") as file:
            func_desc = json.load(file)
    func_to_class = {func: func_class for func_class, funcs in class_to_funcs.items() for func in funcs.keys()}

    if task == "hsd":
        for method in ["baseline", "seen"]:
            if method != "baseline":
                davidson_result = Path(f"./results/{task}/davidson2017/{model_name}_{method}{suffix}.json")
                founta_result = Path(f"./results/{task}/founta2018/{model_name}_{method}{suffix}.json")
            else:
                if not add_examples:
                    davidson_result = Path(f"./results/{task}/davidson2017/{model_name}_baseline_zero.json")
                    founta_result = Path(f"./results/{task}/founta2018/{model_name}_baseline_zero.json")
                else:
                    davidson_result = Path(f"./results/{task}/davidson2017/{model_name}_baseline_example.json")
                    founta_result = Path(f"./results/{task}/founta2018/{model_name}_baseline_example.json")
            if not davidson_result.exists():
                print(f"Generating predictions for davidson2017 with model {model_name} and method {method}")
                all_prompts = prompt("hsd", davidson_test, method=method, func_desc=func_desc, func_to_class=func_to_class,
                            class_to_funcs=class_to_funcs, suite=False, ask_rule=ask_rule, examples=davidson_examples_per_label)
                print(all_prompts[0])
                if chatGPT:
                    responses, all_preds = openai_request(all_prompts, api_token, format_instruction=output_format, max_tokens=max_new_tokens)
                    response_path = "./responses"/davidson_result
                    response_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(response_path, "w") as file:
                        json.dump(responses, file)

                else:
                    all_preds = get_preds(all_prompts, tokenizer, model, data_collator, bs=bs, max_new_tokens=max_new_tokens)
                davidson_result.parent.mkdir(exist_ok=True, parents=True)
                with open(davidson_result, "w") as file:
                    json.dump(all_preds, file)
            else:
                print(f"Results for davidson2017 with model {model_name} already obtained and method {method}")
            if not founta_result.exists():
                print(f"Generating predictions for founta2018 with model {model_name} and method {method}")
                all_prompts = prompt("hsd", founta_test, method=method, func_desc=func_desc, func_to_class=func_to_class,
                            class_to_funcs=class_to_funcs, suite=False, ask_rule=ask_rule, examples=founta_examples_per_label)
                print(all_prompts[0])
                if chatGPT:
                    responses, all_preds = openai_request(all_prompts, api_token, format_instruction=output_format, max_tokens=max_new_tokens)
                    response_path = "./responses"/founta_result
                    response_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(response_path, "w") as file:
                        json.dump(responses, file)
                else:
                    all_preds = get_preds(all_prompts, tokenizer, model, data_collator, bs=bs, max_new_tokens=max_new_tokens)
                founta_result.parent.mkdir(exist_ok=True, parents=True)
                with open(founta_result, "w") as file:
                    json.dump(all_preds, file)
            else:
                print(f"Results for founta2018 with model {model_name} already obtained and method {method}")
    else:
        for method in ["baseline", "seen"]:
            if method != "baseline":
                dataset_result = Path(f"./results/{task}/{dataset_name}/{model_name}_{method}{suffix}.json")
            else:
                if not add_examples:
                    dataset_result =   Path(f"./results/{task}/{dataset_name}/{model_name}_baseline_zero.json")
                else:
                    dataset_result =   Path(f"./results/{task}/{dataset_name}/{model_name}_baseline_example.json")
            if not dataset_result.exists():
                print(f"Generating predictions for {dataset_name} with model {model_name} and method {method}")
                all_prompts = prompt(task, dataset_test, method=method, func_desc=func_desc, func_to_class=func_to_class,
                            class_to_funcs=class_to_funcs, suite=False, ask_rule=ask_rule, examples=examples_per_label)
                print(all_prompts[0])
                if chatGPT:
                    responses, all_preds = openai_request(all_prompts, api_token, format_instruction=output_format, max_tokens=max_new_tokens)
                    response_path = "./responses"/dataset_result
                    response_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(response_path, "w") as file:
                        json.dump(responses, file)
                else:
                    all_preds = get_preds(all_prompts, tokenizer, model, data_collator, bs=bs, max_new_tokens=max_new_tokens)
                dataset_result.parent.mkdir(exist_ok=True, parents=True)
                with open(dataset_result, "w") as file:
                    json.dump(all_preds, file)
            else:
                print(f"Results for {dataset_name} with model {model_name} and method {method} already obtained")
    if methods == "all":
        methods = ["seen", "funcOut", "classOut"]
    else:
        methods = [methods]
    for method in ["baseline"] + methods:
        if method != "baseline":
            result =  Path(f"./results/{task}/suite/{model_name}_{method}{suffix}.json")
        else:
            if not add_examples:
                result =  Path(f"./results/{task}/suite/{model_name}_baseline_zero.json")
            else:
                result =  Path(f"./results/{task}/suite/{model_name}_baseline_example.json")
        if not result.exists():
            print(f"Generating predictions for suite ({method}) with mode {model_name}")
            all_prompts = prompt(task, suite_test, method=method, func_desc=func_desc, func_to_class=func_to_class,
                        class_to_funcs=class_to_funcs, suite=True, ask_rule=ask_rule, examples=examples_per_label)
            print(all_prompts[0])
            if chatGPT:
                responses, all_preds = openai_request(all_prompts, api_token, format_instruction=output_format, max_tokens=max_new_tokens)
                response_path = "./responses"/result
                response_path.parent.mkdir(exist_ok=True, parents=True)
                with open(response_path, "w") as file:
                    json.dump(responses, file)
            else:
                all_preds = get_preds(all_prompts, tokenizer, model, data_collator, bs=bs, max_new_tokens=max_new_tokens)
            result.parent.mkdir(exist_ok=True, parents=True)
            with open(result, "w") as file:
                json.dump(all_preds, file)
        else:
            print(f"Results for suite ({method}) with model {model_name} already obtained")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get the predictions for a task and prompt configuration.')
    parser.add_argument('task', help='The training task.',
                        type=str, choices=["sa", "pi", "rc", "hsd"])
    parser.add_argument('--model_url', help='The model to be prompted.', type=str)
    parser.add_argument('--chatGPT', help='Get preds from chatgpt.', type=str,
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ask_rule', help='Prompt asks for the relevant rules in addition to the task prediction.', type=str,
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--methods', help='The method for including functionality information.', type=str, choices=[
                    "seen", "funcOut", "classOut", "all"], default="all")
    parser.add_argument('--add_examples', help='Add labelled task examples to the prompt', type=str,
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--from_chatGPT', help='Use chatGPT-generated functionality descriptions', type=str,
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--bs', help='Prompts per batch', type=int, default=128)
    args = parser.parse_args()
    main(args.task, args.model_url, args.bs, args.chatGPT, args.methods, args.ask_rule, args.add_examples, args.from_chatGPT)
