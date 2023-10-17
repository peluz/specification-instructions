import numpy as np
from tqdm.auto import tqdm
import pickle
from data_sets.data_utils import extract_perturb, get_id_to_samples
import json
from datasets import Dataset
from random import sample
import pandas as pd
from utils.task_config import task_configs_dataset, task_configs_suite
from utils.util import initialize_seeds


def prompt(task, data, method, func_desc, func_to_class, class_to_funcs, suite=False, ask_rule=False, examples=None):
    ask = ""
    placeholder = ""
    configs = task_configs_dataset[task] if not suite else task_configs_suite[task]
    x_col = configs["x_col"]
    options = configs["options"]
    preamble = configs["preamble"]
    if examples is None:
        prompt_type = "zero"
    else:
        prompt_type = "example"
        initialize_seeds()
    task_prompt = configs[prompt_type]["task_prompt"]
    label_mapping = configs["label_mapping"]
    inputs_prefix = configs["inputs_prefix"]
    targets_prefix = configs["targets_prefix"]
    x_y_delimiter = configs["x_y_delimiter"]
    example_separator = configs["example_separator"]
    final_suffix = configs["final_suffix"]
   
    if ask_rule:
        if task != "rc":
            ask = "\nSelect the most relevant rules, briefly explain how they apply and output exactly the correct option in a new line."
        else:
            ask = "\nSelect the most relevant rules, briefly explain how they apply and output a concise, minimal answer in a new line."
        placeholder = "\n{rule list} and {rationale} indicate where the list of relevant rules and the rationale behind the selection should be generated."
    if prompt_type == "zero":
        if method == "baseline":
            return [f'{zero_shot_pattern(e, task_prompt, "baseline", func_desc, func_to_class, options, x_col, task=task)}' for e in tqdm(data)]
        return [f'{preamble}{get_rules(e, func_desc, method, func_to_class, class_to_funcs)}\n{zero_shot_pattern(e, task_prompt, "baseline", func_desc, func_to_class, options, x_col, task=task)}{ask}' for e in tqdm(data)]
    else:
        if method == "baseline":
            return [few_shot_pattern(e, sample_examples(examples, task, "baseline", e, None, None), task_prompt, label_mapping, inputs_prefix=inputs_prefix, targets_prefix=targets_prefix, x_y_delimiter=x_y_delimiter, example_separator=example_separator, final_suffix=final_suffix, options=options, task=task) for e in tqdm(data)]
        return [f'{preamble}{get_rules(e, func_desc, method, func_to_class, class_to_funcs)}{ask}{placeholder}\n{few_shot_pattern(e, sample_examples(examples, task, "baseline", e, None, None), task_prompt, label_mapping, inputs_prefix=inputs_prefix, targets_prefix=targets_prefix, x_y_delimiter=x_y_delimiter, example_separator=example_separator, final_suffix=final_suffix, options=options, task=task, ask_rule=ask_rule)}' for e in tqdm(data)]



def sa_prompt_control(data, method="baseline", prompt_type="zero", suite=False, **kwargs):
    x_col = "sentence" if not suite else "test_case"
    inputs = data[x_col]
    func_desc=None
    class_to_funcs=None
    perturb_ids=None
    if method != "match":
        class_to_funcs = kwargs["class_to_funcs"]
    options = "OPTIONS:\n- negative\n- positive" if not suite else "OPTIONS:\n- negative\n- positive\n- neutral"
    if prompt_type == "example":
        label_mapping = ["negative", "neutral", "positive"] if method != "baseline" else  ["negative", "positive"]
        if not suite:
            task_prompt = "Is the sentiment of the following sentence positive or negative?"
        else:
            task_prompt = "Is the sentiment of the following sentence positive, negative or neutral?"
        if method != "baseline":
            perturb_ids = kwargs["perturb_ids"]
        return [few_shot_pattern(e, sample_examples(kwargs["examples_per_label"], "sa", method, e, perturb_ids, class_to_funcs), task_prompt, label_mapping, inputs_prefix="Question:\n", targets_prefix="Answer:\n", x_y_delimiter="\n", example_separator="\n\n", final_suffix="", options=options, task="sa") for e in tqdm(data)]
    else:
        if not suite:
            task_prompt = "Is the sentiment of the following sentence positive or negative (see options at the end)?"
        else:
            task_prompt = "Is the sentiment of the following sentence positive, negative or neutral (see options at the end)?"
        if method != "baseline":
            func_desc = kwargs["func_descriptions"]
        return [zero_shot_pattern(e, task_prompt, method, func_desc, class_to_funcs, options, x_col, task="sa") for e in tqdm(data)]

def pi_prompt_control(data, method="baseline", prompt_type="zero", suite=False, **kwargs):
    func_desc=None
    class_to_funcs=None
    perturb_ids=None
    if method != "match":
        class_to_funcs = kwargs["class_to_funcs"]
    options = "OPTIONS:\n- no\n- yes"
    if prompt_type == "example":
        task_prompt = "Are these two questions asking the same thing?"
        label_mapping = ["no", "yes"]
        if method != "baseline":
            perturb_ids = kwargs["perturb_ids"]
        return [few_shot_pattern(e, sample_examples(kwargs["examples_per_label"], "pi", method, e, perturb_ids, class_to_funcs), task_prompt, label_mapping, inputs_prefix="QUES:\n", targets_prefix="ANS:\n", x_y_delimiter="\n\n", example_separator="\n\n\n", final_suffix="", options=options, task="pi") for e in tqdm(data)]
    else:
        task_prompt = "Do those questions have the same meaning?"
        if method != "baseline":
            func_desc = kwargs["func_descriptions"]
        return [zero_shot_pattern(e, task_prompt, method, func_desc, class_to_funcs, options, task="pi") for e in tqdm(data)]


def rc_prompt_control(data, method="baseline", prompt_type="zero", suite=False, **kwargs):
    func_desc=None
    class_to_funcs=None
    perturb_ids=None
    if method != "match":
        class_to_funcs = kwargs["class_to_funcs"]
    task_prompt = "Answer a question about this article:"
    if prompt_type == "example":
        if method != "baseline":
            perturb_ids = kwargs["perturb_ids"]
        return [few_shot_pattern(e, sample_examples(kwargs["examples_per_label"], "rc", method, e, perturb_ids, class_to_funcs), task_prompt, label_mapping=None, inputs_prefix="The problem: ", targets_prefix="The answer: ", x_y_delimiter="\n****\n", example_separator="\n\n\n", final_suffix="", options=None, task="rc") for e in tqdm(data)]
    else:
        if method != "baseline":
            func_desc = kwargs["func_descriptions"]
        return [zero_shot_pattern(e, task_prompt, method, func_desc, class_to_funcs, task="rc", y_col="answers") for e in tqdm(data)]

def hsd_prompt_control(data, method="baseline", prompt_type="zero", suite=False, **kwargs):
    x_col = "text"
    y_col = "label"
    func_desc=None
    class_to_funcs=None
    perturb_ids=None
    if method != "match":
        class_to_funcs = kwargs["class_to_funcs"]
    options = "OPTIONS:\n- no\n- yes"
    if prompt_type == "example":
        label_mapping = ["no", "yes"]
        task_prompt = "Does the following sentence contain hateful language?"
        return [few_shot_pattern(e, sample_examples(kwargs["examples_per_label"], "hsd", method, e, perturb_ids, class_to_funcs), task_prompt, label_mapping, inputs_prefix="Question:\n", targets_prefix="Answer:\n", x_y_delimiter="\n", example_separator="\n\n", final_suffix="", options=options, task="hsd") for e in tqdm(data)]
    else:
        task_prompt = "Does the following sentence contain hateful language (see options at the end)?"
        if method != "baseline":
            func_desc = kwargs["func_descriptions"]
        return [zero_shot_pattern(e, task_prompt, method, func_desc, class_to_funcs, options, x_col, y_col, task="hsd") for e in tqdm(data)]

def get_rules(input, func_desc, method, func_to_class, class_to_funcs):
    n_func = len(func_desc)
    if method == "seen":
        func_desc  =[desc for desc in func_desc.values()]
        assert len(func_desc) == n_func
    else:
        input_func = input["functionality"]
        input_class = func_to_class[input_func]
        if method == "funcOut":
            func_desc = [desc for func, desc in func_desc.items() if func != input_func]
            assert len(func_desc) == n_func - 1
        elif method == "classOut":
            func_desc = [desc for func, desc in func_desc.items() if func_to_class[func] != input_class]
            assert len(func_desc) == n_func - len(class_to_funcs[input_class])
    return "\n".join([f"{num + 1}. {desc}"for num, desc in enumerate(func_desc)])

def zero_shot_pattern(input, task_prompt, method, func_desc=None, class_to_funcs=None, options=None, x_col="test_case", y_col="label", task=None):
    if method == "baseline":
        if task in ["sa", "hsd"]:
            return f"{task_prompt}\n{input[x_col]}\n{options}"
        elif task == "rc":
            return f"{task_prompt}\n{input['context']}\n{input['question']}"
        else:
            return f"{input['question1']}\n{input['question2']}\n{task_prompt}\n{options}"
    if method in ["match", "mismatch"]:
        if method == "match":
            func = func_desc[input["functionality"]]
        else:
            func = func_desc[sample_wrong_func(input, class_to_funcs, y_col=y_col, task=task)]
        if task in ["sa", "hsd"]:
            return f"{task_prompt}\n{input[x_col]}\nConsider that {func}.\n{options}"
        elif task == "rc":
            return f"{task_prompt}\n{input['context']}\n{input['question']}\nConsider that {func}."
        else:
            return f"{input['question1']}\n{input['question2']}\n{task_prompt}\nConsider that {func}.\n{options}"
    if method == "mixed":
        right_func = func_desc[input["functionality"]]
        wrong_func = func_desc[sample_wrong_func(input, class_to_funcs,y_col=y_col, task=task)]
        funcs = [right_func, wrong_func]
        np.random.shuffle(funcs)
        if task in ["sa", "hsd"]:
            return f"{task_prompt}\n{input[x_col]}\nConsider that {', and that '.join(funcs)}.\n{options}"
        elif task == "rc":
            return f"{task_prompt}\n{input['context']}\n{input['question']}\nConsider that {', and that '.join(funcs)}."
        else:
            return f"{input['question1']}\n{input['question2']}\n{task_prompt}\nConsider that {', and that '.join(funcs)}.\n{options}"

def few_shot_pattern(input, examples, task_prompt, label_mapping, inputs_prefix, targets_prefix, x_y_delimiter, example_separator, final_suffix, options=None, task=None, ask_rule=False):
    exemplars = []
    if ask_rule:
        ask = "Rules: {rule list}\nExplanation: {rationale}\n"
    else:
        ask = ""
    if task == "sa":
        try:
            input = input["sentence"]
        except KeyError:
            input = input["test_case"]
    elif task == "pi":  
        input = (input["question1"], input["question2"])
    elif task == "hsd":
        input = input["text"]
    elif task == "rc":
        input = (input["context"], input["question"])
    for example in examples:
        if task == "sa":
            try:
                x, y = example["sentence"], label_mapping[int(example["label"])]
            except KeyError:
                x, y = example["test_case"], label_mapping[int(example["label"])]
        elif task == "pi":
            x, y = (example["question1"], example["question2"]), label_mapping[int(example["label"])]
        elif task == "hsd":
            x, y = example["text"], label_mapping[int(example["label"])]
        elif task == "rc":
            x, y = (example["context"], example["question"]), np.random.choice(example["answers"]["text"])
        exemplars.append(
            f"{inputs_prefix}{form_input(task, x, task_prompt, options)}{x_y_delimiter}{ask}{targets_prefix}{y}{example_separator}")
    if ask_rule:
        targets_prefix = "Rules:"
    return "".join(exemplars) + f"{inputs_prefix}{form_input(task, input, task_prompt, options)}{x_y_delimiter}{targets_prefix}" + final_suffix

def form_input(task, input, task_prompt, options):
    if task in ["sa", "hsd"]:
        return f"{task_prompt} {input}\n{options}"
    elif task == "pi":
        return f"First question: {input[0]}\nSecond question: {input[1]}\n{task_prompt}\n{options}"
    elif task == "rc":
        return f"{task_prompt}\n{input[0]}\n{input[1]}"


def sample_examples(samples_per_label, task, method, input, perturb_ids, class_to_funcs):
    samples = []
    if method == "baseline":
        if task != "rc":
            for cat in samples_per_label:
                idxs = sample(range(cat.num_rows),2)
                for idx in idxs:
                    samples.append(cat[int(idx)])
        else:
            idxs = sample(range(samples_per_label.num_rows),4)
            for idx in idxs:
                samples.append(samples_per_label[int(idx)])
        np.random.shuffle(samples)
    else:
        if method != "mixed":
            if method == "match":
                func = input["functionality"]
                type = input["type"] if task != "hsd" else "mft"
                func_and_type = [(func, type, True)]
            elif method == "mismatch":
                func = sample_wrong_func_example(input, class_to_funcs, task) if task != "rc" else sample_wrong_func(input, class_to_funcs, task="rc")
                type = samples_per_label[func][0]["type"] if task != "hsd" else "mft"
                func_and_type = [(func, type, False)]
            n=4
        else:
            right_func = input["functionality"]
            right_type = input["type"] if task != "hsd" else "mft"
            wrong_func = sample_wrong_func_example(input, class_to_funcs, task) if task != "rc" else sample_wrong_func(input, class_to_funcs, task="rc")
            wrong_type = samples_per_label[wrong_func][0]["type"] if task != "hsd" else "mft"
            func_and_type = [(right_func, right_type, True),
                             (wrong_func, wrong_type, False)]
            np.random.shuffle(func_and_type)
            n=2
        input_label = input["label"] if task != "rc" else None
        if task == "sa":
            if "functionality" in input:
                input_label = input["label"]
            else:
                input_label = input["label"] if input["label"] == 0 else 2
        for func, type, right_func in func_and_type:
            sample_pool = samples_per_label[func]
            if type in ["dir", "inv"]:
                idxs = sample(range(sample_pool.num_rows),n//2)
                if not right_func and task != "rc":
                    while True:
                        sample_labels = [sample_pool[int(idx)]["label"] for idx in idxs]
                        if len(np.unique(sample_labels)) > 1:
                            break
                        elif input_label is None:
                            break
                        elif input_label == "not_0":
                            if sample_labels[0] == 0:
                                break
                        elif int(input_label) != sample_labels[0]:
                            break
                        idxs = sample(range(sample_pool.num_rows),n//2)
                for idx in idxs:
                    orig_sample = sample_pool[int(idx)]
                    id = orig_sample["test_id"]
                    label = orig_sample["label"] if task != "rc" else orig_sample["answers"]["text"][0]
                    perturb_idx = np.random.randint(0, len(perturb_ids[func][id]))
                    perturb_sample = perturb_ids[func][id][perturb_idx]
                    if task != "rc":
                        perturb_sample["label"] = label
                    else:
                        perturb_sample["answers"]["text"] = [label]
                    pair = [orig_sample, perturb_sample]
                    np.random.shuffle(pair)
                    samples.extend(pair)
            else:
                idxs = sample(range(sample_pool.num_rows),n)
                if not right_func and task != "rc":
                    while True:
                        sample_labels = [sample_pool[int(idx)]["label"] for idx in idxs]
                        if len(np.unique(sample_labels)) > 1:
                            break
                        elif input_label is None:
                            break
                        elif input_label == "not_0":
                            if sample_labels[0] == 0:
                                break
                        elif int(input_label) != sample_labels[0]:
                            break
                        idxs = sample(range(sample_pool.num_rows),n)
                for idx in idxs:
                    samples.append(sample_pool[int(idx)])
    return samples

def sample_examples_rule(samples_per_func, task, input, perturb_ids, n_examples=6):
    samples = []
    func = input["functionality"]
    type = input["type"] if task != "hsd" else "mft"
    samples_func = samples_per_func[func]
    if task != "rc":
        labels = np.unique(samples_func["label"])
        sample_pools = [samples_func.filter(lambda x: x["label"] == label) for label in labels]
    else:
        sample_pools = [samples_func]
    n = n_examples // len(sample_pools)
    for sample_pool in sample_pools:
        if type in ["dir", "inv"]:
            idxs = sample(range(sample_pool.num_rows),n)
            for idx in idxs:
                orig_sample = sample_pool[int(idx)]
                id = orig_sample["test_id"]
                perturb_idx = np.random.randint(0, len(perturb_ids[func][id]))
                perturb_sample = perturb_ids[func][id][perturb_idx]
                pair = [orig_sample, perturb_sample]
                samples.append(pair) if not (task == "pi" and type == "dir") else samples.append(pair[-1])
        else:
            idxs = sample(range(sample_pool.num_rows),n)
            for idx in idxs:
                samples.append(sample_pool[int(idx)])
    np.random.shuffle(samples)
    return samples

def sample_wrong_func_example(input, class_to_funcs, task, y_col="label"):
    try:
        if task != "hsd":
            input_func_class = input["capability"]
        else:
            input_func_class = input["functionality"].split("_")[0]
        input_func = input["functionality"]
        label = input["label"]
        if np.random.rand(1) > .5 and (label is None or (label == "not_0" and len([k for k, v in class_to_funcs[input_func_class].items() if len(v) > 1 or v[0] !=0])) or len([k for k, v in class_to_funcs[input_func_class].items() if len(v) > 1 or v[0] !=int(label)]))  :
                funcs = class_to_funcs[input_func_class].copy()
                funcs.pop(input_func)
        else: # Other classes
            funcs = {}
            for cls, dic in class_to_funcs.items():
                if cls != input_func_class:
                    funcs.update(dic)
    except KeyError: # a dataset sample
        if task == "sa":
            label = input[y_col] if input[y_col] == 0 else 2
        elif task == "rc":
            label = None
        else:
            label = input[y_col]
        funcs = {}
        for cls, dic in class_to_funcs.items():
            funcs.update(dic)
    # Filter bad funcs
    if label is None:
        return np.random.choice([func for func, func_labels in funcs.items()])
    elif label == "not_0":
        return np.random.choice([func for func, func_labels in funcs.items() if len(func_labels) > 1 or func_labels[0] == 0])
    else:
        return np.random.choice([func for func, func_labels in funcs.items() if len(func_labels) > 1 or func_labels[0] != int(label)])

def sample_wrong_func(input, class_to_funcs, y_col="label", task=None):
    try:
        if task != "hsd":
            input_func_class = input["capability"]
        else:
            input_func_class = input["functionality"].split("_")[0]
        input_func = input["functionality"]
        input_func_label = class_to_funcs[input_func_class][input_func]
        if np.random.rand(1) > .5 and len([k for k, v in class_to_funcs[input_func_class].items() if v != input_func_label]): # Same class
            funcs = class_to_funcs[input_func_class].copy()
            funcs.pop(input_func)
        else: # Other classes
            funcs = {}
            for cls, dic in class_to_funcs.items():
                if cls != input_func_class:
                    funcs.update(dic)
    except KeyError: # a dataset sample
        if task == "sa":
            input_func_label = input[y_col] if input[y_col] == 0 else 2
        elif task == "rc":
            input_func_label = None
        else:
            input_func_label = input[y_col]
        funcs = {}
        for cls, dic in class_to_funcs.items():
            funcs.update(dic)
    if input_func_label is None:
        return np.random.choice(list(funcs))
    else:
        return np.random.choice([func for func, label in funcs.items() if label != input_func_label])
    
def get_prompts(dataset_train, suite_train, dataset_test, suite_test, model_name, task, method, prompt_type, suite=False, local=True):
    class_to_funcs = None
    func_desc=None
    perturb_ids = None
    examples_per_label=None
    if method != "match":
        with open(f"./data/{task}/suite/class_to_funcs.pkl", "rb") as file:
            class_to_funcs = pickle.load(file)
    if prompt_type == "zero":
        if method != "baseline":
            with open(f"./data/{task}/suite/func_desc.pkl", "rb") as file:
                func_desc = pickle.load(file)
    else:
        if task != "rc" and method == "baseline":
            examples_per_label = [dataset_train.filter(lambda x: x["label"] == label) for label in [0, 1]]
        if method != "baseline":
            if task != "hsd":
                orig_train, perturbs_train = extract_perturb(suite_train)
                with open(f"./results/orig_preds/{task}/{model_name}.json", "r") as file:
                    orig_preds = json.load(file)
                orig_train = orig_train.to_pandas()
                if task != "rc":
                    if task == "sa":
                        label_to_int = {"negative": 0, "neutral": 1, "positive": 2}
                    elif task == "pi":
                        label_to_int = {"no": 0, "yes": 1}
                    orig_train.loc[orig_train["label"].isnull(), "label"] = [label_to_int[pred] for pred in orig_preds]
                    orig_train.loc[orig_train["label"] == "not_0", "label"] = 1
                    orig_train["label"] = pd.to_numeric(orig_train["label"])
                else:
                    orig_train.loc[orig_train.type == "inv", "answers"] = [{"text": [pred]} for pred in orig_preds]
                perturb_ids = get_id_to_samples(perturbs_train)
                funcs = orig_train.functionality.unique()
                orig_train = Dataset.from_pandas(orig_train)
                examples_per_label = {}
                for func in funcs:
                    samples = orig_train.filter(lambda x: x["functionality"] == func)
                    examples_per_label[func] = samples
                    func_class = samples[0]["capability"]
                    if method != "match" and task != "rc":
                        class_to_funcs[func_class][func] = np.unique(samples["label"]).astype(int)
            else:
                examples_per_label = {}
                funcs = np.unique(suite_train["functionality"])
                for func in funcs:
                    samples = suite_train.filter(lambda x: x["functionality"] == func)
                    examples_per_label[func] = samples
                    func_class = func.split("_")[0]
                    if method != "match":
                        class_to_funcs[func_class][func] = np.unique(samples["label"]).astype(int)
    if suite:
        test = suite_test
    else:
        test = dataset_test
    if task == "rc" and method == "baseline":
        examples_per_label = dataset_train
    if task == "sa":
        all_prompts = sa_prompt_control(test, "flan", method=method, prompt_type=prompt_type,
                                examples_per_label=examples_per_label, func_descriptions=func_desc,
                                class_to_funcs=class_to_funcs, perturb_ids=perturb_ids, suite=suite)
    elif task == "pi":
        all_prompts = pi_prompt_control(test, "flan", method=method, prompt_type=prompt_type,
                                examples_per_label=examples_per_label, func_descriptions=func_desc,
                                class_to_funcs=class_to_funcs, perturb_ids=perturb_ids, suite=suite)
    elif task == "hsd":
        all_prompts = hsd_prompt_control(test, "flan", method=method, prompt_type=prompt_type,
                                examples_per_label=examples_per_label, func_descriptions=func_desc,
                                class_to_funcs=class_to_funcs, perturb_ids=perturb_ids, suite=suite)
    elif task == "rc":
        all_prompts = rc_prompt_control(test, "flan", method=method, prompt_type=prompt_type,
                                func_descriptions=func_desc, examples_per_label = examples_per_label,
                                class_to_funcs=class_to_funcs, perturb_ids=perturb_ids, suite=suite)
    return all_prompts