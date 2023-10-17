from datasets import Dataset
from models.inference import openai_request
from data_sets.data_utils import get_suite, extract_perturb, get_id_to_samples
from utils.task_config import task_configs_dataset
from utils.prompts import sample_examples_rule
from utils.results import *
from utils.util import initialize_seeds
import json
import pandas as pd
from pathlib import Path
import json
import numpy as np
import config


def get_examples_per_func(task, suite_train):
    if task != "hsd":
        orig_train, perturbs_train = extract_perturb(suite_train)
        with open(f"./results/orig_preds/{task}/flan-t5-xxl.json", "r") as file:
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
        examples_per_func = {}
        for func in funcs:
            samples = orig_train.filter(lambda x: x["functionality"] == func)
            examples_per_func[func] = samples
    else:
        examples_per_func = {}
        funcs = np.unique(suite_train["functionality"])
        for func in funcs:
            samples = suite_train.filter(lambda x: x["functionality"] == func)
            examples_per_func[func] = samples
        perturb_ids = None
    return examples_per_func, perturb_ids

func_name = {
    'counter_quote_nh': 'Denouncements of hate that quote it',
    'counter_ref_nh': "Denouncements of hate that make direct reference to it",
    'derog_dehum_h': "Dehumanisation (explicit)",
    'derog_impl_h': "Implicit derogation",
    'derog_neg_attrib_h': "Description using very negative attributes (explicit)",
    'derog_neg_emote_h': "Expression of strong negative emotions (explicit)",
    'ident_neutral_nh': "Neutral statements using protected group identifiers",
    'ident_pos_nh': "Positive statements using protected group identifiers",
    'negate_neg_nh': "Non-hate expressed using negated hateful statement",
    'negate_pos_h': "Hate expressed using negated positive statement",
    'phrase_opinion_h': "Hate phrased as an opinion",
    'phrase_question_h': "Hate phrased as a question",
    'profanity_h': "Hate expressed using profanity",
    'profanity_nh': "Non-hateful use of profanity",
    'ref_subs_clause_h': "Hate expressed through reference in subsequent clauses",
    'ref_subs_sent_h': "Hate expressed through reference in subsequent sentences",
    'slur_h': "Hate expressed using slur",
    'slur_homonym_nh': "Non-hateful homonyms of slurs",
    'slur_reclaimed_nh': "Reclaimed slurs",
    'spell_char_del_h': "Missing characters",
    'spell_char_swap_h': "Swaps of adjacent characters",
    'spell_leet_h': "Leet speak spellings",
    'spell_space_add_h': "Added spaces between chars",
    'spell_space_del_h': "Missing word boundaries",
    'target_group_nh': "Abuse targeted at nonprotected groups (e.g. professions)",
    'target_indiv_nh': "Abuse targeted at individuals (not as member of a prot. group)",
    'target_obj_nh': "Abuse targeted at objects",
    'threat_dir_h': "Direct threat",
    'threat_norm_h': "Threat as normative statement"    
}

def get_rule_prompts(task, samples_per_func, perturb_ids):
    initialize_seeds()
    if task == "sa":
        label_mapping = ["negative", "neutral", "positive"]
    else:
        label_mapping = task_configs_dataset[task]["label_mapping"]
    funcs = list(samples_per_func.keys())

    prompts = []
    for func in funcs:
        inp = samples_per_func[func][0]
        typ = inp["type"] if task != "hsd" else "mft"
        preamble= f"""Task: {task_configs_dataset[task]["name"]}
Functionality: {func_name[func] if task == "hsd" else func}
Consider the following {task_configs_dataset[task]["format"][typ]}:"""
        n_examples = 6 if not (task == "rc" and typ == "inv") else 2
        examples = sample_examples_rule(samples_per_func, task, inp, perturb_ids, n_examples=n_examples)
        if typ == "mft" or (typ == "dir" and task == "pi"):
            if task in ["sa", "hsd"]:
                demonstrations = [f"Sentence: {e['test_case']}\nLabel: {label_mapping[int(e['label'])]}" for e in examples]
            elif task == "pi":
                demonstrations = [f"Question 1: {e['question1']}\nQuestion 2: {e['question2']}\nLabel: {label_mapping[int(e['label'])]}" for e in examples]
            elif task == "rc":
                demonstrations = [f"Context: {e['context']}\nQuestion: {e['question']}\nAnswer: {e['answers']['text'][0]}" for e in examples]
        else:
            if task == "sa":
                demonstrations = [f"Sentence: {e1['test_case']}\nPerturbation: {e2['test_case']}" for e1, e2 in examples]
            elif task == "pi":
                demonstrations = [f"Question 1: {e1['question1']}\nQuestion 2: {e1['question2']}\nPerturbation 1: {e2['question1']}\nPerturbation 2: {e2['question2']}" for e1, e2 in examples]
            elif task == "rc":
                demonstrations = [f"Context: {e1['context']}\nQuestion: {e1['question']}\nPerturbed context: {e2['context']}\nPerturbed question: {e2['question']}" for e1,e2 in examples]
        demonstration = "\n\n".join(demonstrations)
        if typ == "mft" or (typ == "dir" and task == "pi"):
            instruction = f"""Write a general rule that explains the {task_configs_dataset[task]["label"]} above.
Rule: if"""
        elif typ == "inv":
            instruction = f"""Write a general rule that explains why the perturbations do not change the original {task_configs_dataset[task]["target"]}. Avoid mentioning the perturbations explicitly.
Rule: The perturbations do not change the original {task_configs_dataset[task]["target"]} because if"""
        else:
            instruction = f"""Write a general rule that explains why the perturbations {task_configs_dataset[task]["dir_mapping"][inp["direction"]]}. Avoid mentioning the perturbations explicitly.
Rule: The perturbations {task_configs_dataset[task]["dir_mapping"][inp["direction"]]} because if"""
        prompts.append(f"{preamble}\n\n{demonstration}\n\n{instruction}")
    return prompts

for task in ["sa", "pi", "rc", "hsd"]:
    if task == "sa":
        suite_path = config.sa_path
    elif task == "pi":
        suite_path = config.pi_path
    elif task == "rc":
        suite_path = config.rc_path
    if task != "hsd":
        suite_train = get_suite(suite_path)["train"]
    else:
        suite_train = get_suite(config.hatecheck_path, hateCheck=True).rename_column("label_gold", "label")["train"]
    
    samples_per_func, perturb_ids = get_examples_per_func(task, suite_train)
    prompts = get_rule_prompts(task, samples_per_func, perturb_ids)
    print(prompts[0])
    api_token = config.openai_key
    output_format = ""
    max_new_tokens=150
    path = Path(f"./data/chatGPTgeneratedRules/{task}/rules.json")
    if path.exists():
        print(f"Already generated rules for {task}")
        continue
    responses, all_preds = openai_request(prompts, api_token, format_instruction=output_format, max_tokens=max_new_tokens)
    path.parent.mkdir(exist_ok=True, parents=True)
    funcs = list(samples_per_func.keys())
    all_preds = {func: "If " + pred for func, pred in zip(funcs, all_preds)}
    with open(path, "w") as file:
        json.dump(all_preds, file)
    response_path = Path(f"./responses/{task}_rules.json")
    response_path.parent.mkdir(exist_ok=True, parents=True)
    with open(response_path, "w") as file:
        json.dump(responses, file)