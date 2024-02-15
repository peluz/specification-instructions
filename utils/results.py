import json
import pandas as pd
import numpy as np
import config
from scipy.stats import binomtest, hmean
from nltk.tokenize import sent_tokenize
import collections
import re
import string
from sklearn.metrics import f1_score
import random
from tqdm.auto import tqdm
from copy import deepcopy
from pathlib import Path
from collections import Counter

model_order = ["small", "base", "large", "xl", "xxl", "beta" "chatGPT"]

score_order = ["baseline", "Task", "Task+Rules", "Task+Spec"]

add_order = ["", "+Ex", "(chatGPT)+Ex", "+Rat", "+Ex+Rat"]

method_order = [score + add for score in score_order for add in add_order]

order = {x: i for i, x in enumerate(model_order + method_order)}



def load_results(path, hatecheck=False, file_type="json"):
    if not hatecheck:
        results = {}
        for children in path.rglob(f"*{file_type}"):
            if "hits" in children.name:
                continue
            model = children.name[:-(len(file_type)+1)]
            if file_type == "json":
                with open(children, "r") as file:
                    results[model] = json.load(file)
            elif file_type == "csv":
                results[model] = pd.read_csv(children, index_col=0)
    else:
        results = {}
        hits = {}
        hatecheck_test = pd.read_csv(config.hatecheck_path/"hatecheck_test.csv", index_col=0)
        funcs = hatecheck_test.functionality.tolist() 
        for children in path.rglob(f"*{file_type}"):
            if "hits" in children.name:
                continue
            model = children.name[:-(len(file_type)+1)]
            with open(children, "r") as file:
                    preds = json.load(file)
                    # if "rules" in model:
                    #     preds = [1 if ("yes" in pred.lower()) or ("hateful" in pred.split()[-1].lower() and "not" not in pred.split()[-2]) else 0 for pred in preds]
                    # else:
                    #     preds = [1 if "yes" in pred.lower() else 0 for pred in preds]
                    try:
                        preds = [extract_answer(pred.lower(), ["no", "yes"], "hsd") for pred in preds]
                    except AttributeError:
                        preds = [extract_answer(pred[0]["generated_text"].lower(), ["no", "yes"], "hsd") for pred in preds]
            hits = pd.DataFrame.from_dict({"funcs": funcs, "hits": (np.array(hatecheck_test["label_gold"])== np.array(preds)).astype(int), "labels": hatecheck_test["label_gold"]})
            hits_path = Path(f"{path}/{model}_hits.json")
            if not hits_path.exists():
                all_hits =  {f: hits[hits["funcs"] == f]["hits"].tolist() for f in funcs}
                with open(hits_path, "w") as file:
                    json.dump(all_hits, file)
            results[model] = pd.DataFrame.from_dict({model: hits.groupby("funcs").mean().to_dict()["hits"]}, orient="index")
            results[model]["avg"] = results[model].mean(axis=1)
    return results

def load_hits(path):
    results = {}
    for children in path.rglob("*hits.json"):
        model = children.name[:-(len("_hits.json"))]
        with open(children, "r") as file:
            results[model] = json.load(file)
    return results


def extract_answer_fallback(generation, option_range):
    options = re.findall(rf'({option_range})', generation)
    if len(options) == 0: return option_range.split("|")[0].replace(r"\b", "")
        # choice = float(input(generation))
        # print("============================")
        # return choice[0], 1
    options = Counter(options).most_common(2)
    return options[0][0]

def extract_answer(generation, options, task, verbose=False):
    generation = generation.strip()
    option_range = "|".join([rf"\b{x}\b" for x in options])
    only = re.search(rf'^({option_range})$', generation)
    answer = re.search(rf'(answer|output|ans) *(is)*:*[\n ]*\"*({option_range})', generation)
    starts_with = re.search(rf'^({option_range})', generation)
    ends_with =  re.search(rf'({option_range})\"*\.*$', generation)
    if answer:
        return options.index(answer.groups()[-1].split()[-1].lstrip("\""))
    elif only:
        return options.index(only.groups()[-1])
    elif starts_with:
        return options.index(starts_with.groups()[-1].rstrip(".,\n)"))
    elif ends_with:
        return options.index(ends_with.groups()[0])
    else:
        if task == "pi":
            not_the_same = re.search(rf'not[^.]*same', generation)
            not_different = re.search(rf'not[^.]*different', generation)
            same = re.search(rf'same', generation)
            different = re.search(rf'different', generation)
            if not_the_same: return 0
            elif not_different: return 1
            elif same: return 1
            elif different: return 0
        if verbose:
            print(generation)
            print("==============")
        return options.index(extract_answer_fallback(generation, option_range))
    
def extract_squad_answer(pred):
    pred = pred.lower().strip()
    answer_template = re.search(rf"the answer: (.*)", pred)
    if answer_template:
        return  answer_template.groups()[-1]
    pred_segmented = pred.split("\n")
    if len(pred_segmented) >=3: pred = "\n".join([x for x in pred_segmented if "rule list:" not in x and "explanation:" not in x])
    answer_fallback = re.search(rf"((the *)|^|\n)(answer|output) *(is)*:*[\n ]*\"*(.+)", pred)
    if answer_fallback:
        return  answer_fallback.groups()[-1]
    else:
        sentences = sent_tokenize(pred)
        if len(pred) == 0:
            return ""
        else:
            return sentences[-1] if "rule" in sentences[0].lower() else sentences[0]

def get_dataset_scores(task, results, labels, metric, with_preds=True):
    dataset_scores = {}
    dataset_preds = {}
    for model, all_preds in tqdm(results.items()):
        if type(all_preds[0]) in (str, np.str_):
            preds = all_preds
        else:
            try:
                preds = [pred["generated_text"] for pred in all_preds]
            except KeyError:
                preds = [pred["text"] for pred in all_preds]
            except TypeError:
                preds = [pred[0]["generated_text"] for pred in all_preds]
        if task == "sa":
            # if "rules" in model:
            #     preds = [pred.split()[-1] for pred in preds]
            # preds = [1 if "positive" in pred.lower() else 0 for pred in preds]
            preds = [extract_answer(pred.lower(), ["negative", "positive"], task) for pred in preds]
            references = labels
        elif task in ["hsd", "pi"]:
            # if "rules" in model:
            #     if task == "pi":
            #         preds = [pred.split()[0] if ("yes" in pred.split()[0].lower() or "no" in pred.split()[0].lower()) else pred.split()[-1] for pred in preds]
            #         preds = [1 if "yes" in pred.lower() else 0 for pred in preds]
                # elif task == "hsd":
                #     preds = [1 if ("yes" in pred.lower()) or ("hateful" in pred.split()[-1].lower() and "not" not in pred.split()[-2]) else 0 for pred in preds]
            # else:
                # preds = [1 if "yes" in pred.lower() else 0 for pred in preds]
            preds = [extract_answer(pred.lower(), ["no", "yes"], task) for pred in preds]
            references = labels
        elif task == "rc":
            processed_path = Path(f"./results/rc/squad/{model}_processed.json")
            if processed_path.exists():
                preds = json.load(open(processed_path, "r"))
            # if "rules" in model:
            #     processed_preds = []
            #     for pred in preds:
            #         if "example" in model:
            #             try:
            #                 processed_preds.append(re.search("(?<=The answer: ).+"  , pred).group())
            #             except AttributeError:
            #                 sentences = sent_tokenize(pred)
            #                 if len(pred) == 0:
            #                     processed_preds.append("")
            #                 else:
            #                     processed_preds.append(sentences[-1])
            #         else:
            #             sentences = sent_tokenize(pred)
            #             if len(pred) == 0:
            #                 processed_preds.append("")
            #             else:
            #                 processed_preds.append(sentences[-1] if "rule" in sentences[0].lower() else sentences[0])
            #         preds = processed_preds
            else:
                preds = [{"prediction_text": extract_squad_answer(pred), "id": i} for i, pred in enumerate(preds)]
                # json.dump(preds, open(processed_path, "w"))
            references  = [{"answers": answer, "id": i} for i, answer in enumerate(labels)]
        dataset_scores[model] = metric.compute(predictions=preds, references=references)
        dataset_preds[model] = preds
    return dataset_scores, dataset_preds if with_preds else dataset_scores

def get_suite_preds(results, task):
    suite_preds = {}
    for model, all_preds in tqdm(results.items()):
        if type(all_preds[0]) == str:
            preds = all_preds
        else:
            try:
                preds = [pred["generated_text"] for pred in all_preds]
            except KeyError:
                preds = [pred["text"] for pred in all_preds]
            except TypeError:
                preds = [pred[0]["generated_text"] for pred in all_preds]
        if task == "sa":
            # if "rules" in model:
            #     preds = [pred.split()[-1] for pred in preds]
            # suite_preds[model] = [2 if "positive" in pred.lower() else (1 if "neutral" in pred.lower() else 0) for pred in preds]
            suite_preds[model] = [extract_answer(pred.lower(), ["negative", "neutral", "positive"], task) for pred in preds]
        elif task == "rc":
            # if "rules" in model:
            #     processed_preds = []
            #     for pred in preds:
            #         if "example" in model:
            #             try:
            #                 processed_preds.append(re.search("(?<=The answer: ).+"  , pred).group())
            #             except AttributeError:
            #                 sentences = sent_tokenize(pred)
            #                 if len(pred) == 0:
            #                     processed_preds.append("")
            #                 else:
            #                     processed_preds.append(sentences[-1])
            #         else:
            #             sentences = sent_tokenize(pred)
            #             if len(pred) == 0:
            #                 processed_preds.append("")
            #             else:
            #                 processed_preds.append(sentences[-1] if "rule" in sentences[0].lower() else sentences[0])
            #     preds = processed_preds
            preds = [extract_squad_answer(pred) for pred in preds]
            suite_preds[model] = [normalize_answer(pred) for pred in preds]
        elif task == "pi":
            # if "rules" in model:
            #     preds = [pred.split()[0] if ("yes" in pred.split()[0].lower() or "no" in pred.split()[0].lower()) else pred.split()[-1] for pred in preds]
            # suite_preds[model] = [1 if "yes" in pred.lower() else 0 for pred in preds]
            suite_preds[model] = [extract_answer(pred.lower(), ["no", "yes"], task) for pred in preds]
        else:
            raise NotImplementedError
    return suite_preds


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score_squad(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)    

def get_significance(task, preds_1, preds_2, labels):
    if task != "rc":
        corrects1 = (np.array(preds_1) == np.array(labels)).astype(int)
        corrects2 = (np.array(preds_2) == np.array(labels)).astype(int)
    else:
        corrects1 = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds_1, labels)]).astype(int)
        corrects2 = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds_2, labels)]).astype(int)
    diffs = corrects2 - corrects1
    successes = (diffs == 1).sum()
    trials = np.abs(diffs).sum()
    return binomtest(successes, trials)

def get_significance_hsd(model1, model2, labels, trials, seed=42):
    score1 = f1_score(labels, model1, average="binary")
    score2 = f1_score(labels, model2, average="binary")
    print('# score(model1) = %f' % score1)
    print('# score(model2) = %f' % score2)

    diff = abs(score1 - score2)
    print('# abs(diff) = %f' % diff)

    uncommon = [i for i in range(len(model1)) if model1[i] != model2[i]]

    better = 0
    
    rng = random.Random(seed)
    getrandbits_func = rng.getrandbits

    for _ in tqdm(range(trials)):
        model1_local, model2_local = list(model1), list(model2)

        for i in uncommon:
            if getrandbits_func(1) == 1:
                model1_local[i], model2_local[i] = model2[i], model1[i]

        assert len(model1_local) == len(model2_local) == len(model1) == len(model2)

        diff_local = abs(f1_score(labels, model1_local, average="macro") - f1_score(labels, model2_local, average="macro"))

        if diff_local >= diff:
            better += 1

    p = (better + 1) / (trials + 1)
    print(f"p_value: {p}, successes: {better}")
    return p

def get_significance_randomized(hits1, hits2, trials, seed=42):
    score1 = np.mean([np.nanmean(v) for v in hits1.values()])
    score2 = np.mean([np.nanmean(v) for v in hits2.values()])
    print('# score(model1) = %f' % score1)
    print('# score(model2) = %f' % score2)

    diff = abs(score1 - score2)
    print('# abs(diff) = %f' % diff)

    uncommon = {k: [i for i in range(len(v)) if v[i] != hits2[k][i]] for k, v in hits1.items()}

    better = 0
    
    rng = random.Random(seed)
    getrandbits_func = rng.getrandbits

    for _ in tqdm(range(trials)):
        suite1, suite2 = deepcopy(hits1), deepcopy(hits2)

        for k in hits1.keys():
            for i in uncommon[k]:
                if getrandbits_func(1) == 1:
                    suite1[k][i], suite2[k][i] = hits2[k][i], hits1[k][i]
            assert len(suite1[k]) == len(suite2[k]) == len(hits2[k]) == len(hits1[k])
        new_score1 = np.mean([np.nanmean(v) for v in suite1.values()])
        new_score2 = np.mean([np.nanmean(v) for v in suite2.values()])

        diff_local = abs(new_score1 - new_score2)

        if diff_local >= diff:
            better += 1

    p = (better + 1) / (trials + 1)
    print(f"p_value: {p}, successes: {better}")
    return p

def get_significance_randomized_g(hits_dataset1, hits_suite1, hits_dataset2, hits_suite2, trials, seed=42, labels=None, f1=False):
    suite_score1 = np.mean([np.nanmean(v) for v in hits_suite1.values()])
    suite_score2 = np.mean([np.nanmean(v) for v in hits_suite2.values()])
    if not f1:
        dataset_score1 = np.nanmean(hits_dataset1)
        dataset_score2 = np.nanmean(hits_dataset2)
    else:
        dataset_score1 = f1_score(labels, hits_dataset1, average="binary")
        dataset_score2 = f1_score(labels, hits_dataset2, average="binary")
    score1 = hmean([suite_score1, dataset_score1])
    score2 = hmean([suite_score2, dataset_score2])
    print('# score(model1) = %f' % score1)
    print('# score(model2) = %f' % score2)

    diff = abs(score1 - score2)
    print('# abs(diff) = %f' % diff)


    uncommon_dataset = [i for i in range(len(hits_dataset1)) if hits_dataset1[i] != hits_dataset2[i]]
    uncommon_suite = {k: [i for i in range(len(v)) if v[i] != hits_suite2[k][i]] for k, v in hits_suite1.items()}

    better = 0
    
    rng = random.Random(seed)
    getrandbits_func = rng.getrandbits

    for _ in tqdm(range(trials)):
        dataset1, dataset2 = list(hits_dataset1), list(hits_dataset2)
        suite1, suite2 = deepcopy(hits_suite1), deepcopy(hits_suite2)
        for i in uncommon_dataset:
            if getrandbits_func(1) == 1:
                dataset1[i], dataset2[i] = hits_dataset2[i], hits_dataset1[i]
        assert len(hits_dataset1) == len(hits_dataset2) == len(dataset1) == len(dataset2)
        
        for k in hits_suite1.keys():
            for i in uncommon_suite[k]:
                if getrandbits_func(1) == 1:
                    suite1[k][i], suite2[k][i] = hits_suite2[k][i], hits_suite1[k][i]
            assert len(suite1[k]) == len(suite2[k]) == len(hits_suite2[k]) == len(hits_suite1[k])
        
        new_suite_score1 = np.mean([np.nanmean(v) for v in suite1.values()])
        new_suite_score2 = np.mean([np.nanmean(v) for v in suite2.values()])
        if not f1:
            new_dataset_score1 = np.nanmean(dataset1)
            new_dataset_score2 = np.nanmean(dataset2)
        else:
            new_dataset_score1 = f1_score(labels, dataset1, average="binary")
            new_dataset_score2 = f1_score(labels, dataset2, average="binary")
        new_score1 = hmean([new_suite_score1, new_dataset_score1])
        new_score2 = hmean([new_suite_score2, new_dataset_score2])
        diff_local = abs(new_score1 - new_score2)

        if diff_local >= diff:
            better += 1

    p = (better + 1) / (trials + 1)
    print(f"p_value: {p}, successes: {better}")
    return p

def get_score(hits_dataset_by_task, hits_suite_by_task_score, hate_labels):
    gscores = {}
    scores = ["seen", "funcOut", "classOut"]
    for task, hits_dataset in hits_dataset_by_task.items():
        if "hsd" not in task:
            dataset_score = np.nanmean(hits_dataset)
        else:
            labels = hate_labels[task]
            dataset_score = f1_score(labels, hits_dataset, average="binary")
        suite_scores = [np.mean([np.nanmean(v) for v in hits_suite.values()]) for hits_suite in hits_suite_by_task_score[task]]
        gscores[task] = {gscore: hmean([dataset_score, suite_score]) for gscore, suite_score in zip(scores, suite_scores)}
    gscores["avg"] = np.mean([x for task_scores in gscores.values() for x in task_scores.values()])
    return gscores


def randomized_test_avg(hits_dataset_by_task1, hits_suite_by_task_score1, hits_dataset_by_task2, hits_suite_by_task_score2, trials, hate_labels, seed=42, verbose=False):
    score_to_idx = {score: k for k, score in enumerate(["seen", "funcOut", "classOut"])}
    g_scores1 = get_score(hits_dataset_by_task1, hits_suite_by_task_score1, hate_labels)
    g_scores2 = get_score(hits_dataset_by_task2, hits_suite_by_task_score2, hate_labels)
    for task, g_scores in g_scores1.items():
        if task == "avg":
            print('# avg score(model1) = %f' % g_scores)
            print()
        else:
            for score, value in g_scores.items():
                    print(f"{score} score(model 1) for task {task} = {value}")
    for task, g_scores in g_scores2.items():
        if task == "avg":
            print('# avg score(model2) = %f' % g_scores)
            print()
        else:
            for score, value in g_scores.items():
                    print(f"{score} score(model 2) for task {task} = {value}")
    
    diffs = {}
    for task, values in g_scores1.items():
        if task == "avg":
            diffs["avg"] = abs(values - g_scores2[task])
            print('# abs avg(diff) = %f' % diffs["avg"])
            print()
        else:
            for score, value in values.items():
                diffs.setdefault(task, {})[score] = abs(value - g_scores2[task][score])
                print(f"abs(diff) for {task} {score} = {diffs[task][score]}")

    uncommon_datasets = {}
    for k, v in hits_dataset_by_task1.items():
        uncommon_datasets[k] = [i for i in range(len(v)) if v[i] != hits_dataset_by_task2[k][i]] 
    uncommon_suites = {task: [{k: [i for i in range(len(v)) if v[i] != hits_suite_by_task_score2[task][n][k][i]] for k, v in result.items()}
                              for n, result in enumerate(results)] for task, results in hits_suite_by_task_score1.items()}
    better = {}
    for task, diff in diffs.items():
        if task == "avg":
            better["avg"] = 0
        else:
            for score, value in diff.items():
                better.setdefault(task, {})[score] = 0
    
    rng = random.Random(seed)
    getrandbits_func = rng.getrandbits

    for _ in tqdm(range(trials)):
        datasets1, datasets2 = deepcopy(hits_dataset_by_task1), deepcopy(hits_dataset_by_task2)
        suite1, suite2 = deepcopy(hits_suite_by_task_score1), deepcopy(hits_suite_by_task_score2)
        for task, uncommons in uncommon_datasets.items():
            for i in uncommons:
                if getrandbits_func(1) == 1:
                    datasets1[task][i], datasets2[task][i] = hits_dataset_by_task2[task][i], hits_dataset_by_task1[task][i]
        
        for task, scores in uncommon_suites.items():
            for n, funcs in enumerate(scores):
                for k, uncommons in funcs.items():
                    for i in uncommons:
                        if getrandbits_func(1) == 1:
                            suite1[task][n][k][i], suite2[task][n][k][i] = hits_suite_by_task_score2[task][n][k][i], hits_suite_by_task_score1[task][n][k][i]

        new_score1 = get_score(datasets1, suite1, hate_labels)
        new_score2 = get_score(datasets2, suite2, hate_labels)

        for task, values in new_score1.items():
            if task == "avg":
                if abs(values - new_score2[task]) >= diffs["avg"]:
                    better["avg"] += 1
                    if verbose:
                        print("New avg scores:")
                        print(values, new_score2[task])
            else:
                for score, value in values.items():
                    if abs(value - new_score2[task][score]) >= diffs[task][score]:
                        better[task][score] += 1
                        if verbose:
                            print(f"delta_score for {task} {score}")
                            print(value - new_score2[task][score])
                            print("delta_dataset:")
                            if "hsd" not in task:
                                print(np.mean(datasets1[task]) - np.mean(hits_dataset_by_task1[task]))
                                print(np.mean(datasets2[task]) - np.mean(hits_dataset_by_task2[task]))
                            else:
                                print(f1_score(hate_labels[task], datasets1[task]) - f1_score(hate_labels[task], hits_dataset_by_task1[task]))
                                print(f1_score(hate_labels[task], datasets2[task]) - f1_score(hate_labels[task],hits_dataset_by_task2[task]))
                            print("delta_suite:")
                            print({k: np.nanmean(v) - np.nanmean(hits_suite_by_task_score1[task][score_to_idx[score]][k]) for k, v in suite1[task][score_to_idx[score]].items()})
                            print({k: np.nanmean(v) - np.nanmean(hits_suite_by_task_score2[task][score_to_idx[score]][k]) for k, v in suite2[task][score_to_idx[score]].items()})
    ps = {}                     
    for task, values in better.items():
        if task == "avg":
            ps["avg"] =  (values + 1) / (trials + 1)
            print(f"p_value: {ps['avg']}, successes: {better['avg']}, avg")
            if ps["avg"] > .05:
                print("Difference not significant")
        else:
            for score, value in values.items():
                ps.setdefault(task, {})[score] = (value + 1) / (trials + 1)
                print(f"p_value: {ps[task][score]}, successes: {better[task][score]} for {task} {score}")
                if ps[task][score] > .05:
                    print("Difference not significant")
    print()
    return ps

def get_pvalues_dataset(preds, task, dataset, test):
    try:
        with open(f"./data/{task}/{dataset}/pvalue.json", "r") as file:
            pvalues = json.load(file)
    except FileNotFoundError:
        pvalues = {}
        if task != "rc":
            labels = test["label"]
        else:
            labels =[x["text"] for x in test["answers"]]
        for model in np.unique([x.split("_")[0] for x in preds.keys()]):
            key = f"{model}_seen"
            print(f"Comparing {key} with baseline")
            try:
                if task != "hsd":
                    sig_test = get_significance(task, preds[f"{model}_baseline_zero"], preds[key], labels)
                    print(sig_test)
                    pvalue = sig_test.pvalue
                else:
                    pvalue = get_significance_hsd(preds[f"{model}_baseline_zero"], preds[key], labels, trials=10000)
                    print(pvalue)
                pvalues[key] = pvalue
                if pvalue > .05:
                    print("Difference is not significant")
            except ValueError:
                print("Identical predictions")
                pvalues[key] = np.inf
        with open(f"./data/{task}/{dataset}/pvalue.json", "w") as file:
            json.dump(pvalues, file)
    return pvalues

def get_pvalues_suite(task, models):
    try:
        with open(f"./data/{task}/suite/pvalue.json", "r") as file:
            pvalues = json.load(file)
    except FileNotFoundError:
        pvalues = {}
        for model in models:
            with open(f"./results/{task}/suite/{model}_baseline_zero_hits.json", "r") as file:
                    hits_baseline = json.load(file)
            for method in ["seen", "funcOut", "classOut"]:
                key = f"{model}_{method}"
                print(f"Comparing {key} with baseline")
                with open(f"./results/{task}/suite/{key}_hits.json", "r") as file:
                    hits = json.load(file)                    
                pvalue = get_significance_randomized(hits_baseline, hits ,trials=10000)
                print(pvalue)
                pvalues[key] = pvalue
                if pvalue > .05:
                    print("Difference is not significant")
        with open(f"./data/{task}/suite/pvalue.json", "w") as file:
            json.dump(pvalues, file)
    return pvalues

def get_pvalues_g(task, preds, models, methods, dataset, test):
    f1 = False
    # try:
    #     if task == "hsd":
    #         with open(f"./data/{task}/{dataset}pvalue.json", "r") as file:
    #             pvalues = json.load(file)
    #     else:
    #         with open(f"./data/{task}/pvalue.json", "r") as file:
    #             pvalues = json.load(file)
    # except FileNotFoundError:
    pvalues = {}
    if task != "rc":
        labels = test["label"]
        labels = np.array(labels)
    else:
        labels =[x["text"] for x in test["answers"]]
    for model in models:
        if task  =="hsd":
            baseline_hits_dataset_zero = preds[f"{model}_baseline_zero"]
            baseline_hits_dataset_example = preds[f"{model}_baseline_example"]
            f1 = True
        elif task == "rc":
            baseline_hits_dataset_zero = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds[f"{model}_baseline_zero"], labels)]).astype(int)
            baseline_hits_dataset_example = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds[f"{model}_baseline_example"], labels)]).astype(int)
        else:
            baseline_hits_dataset_zero = (preds[f"{model}_baseline_zero"] == labels).astype(int)
            baseline_hits_dataset_example = (preds[f"{model}_baseline_example"] == labels).astype(int)
        with open(f"./results/{task}/suite/{model}_baseline_zero_hits.json", "r") as file:
            baseline_hits_suite_zero = json.load(file)
        with open(f"./results/{task}/suite/{model}_baseline_example_hits.json", "r") as file:
            baseline_hits_suite_example = json.load(file)     
        for method in methods:
            suffix = f"_{method}" if len(method) > 0 else method
            if task == "hsd":
                method_hits_dataset = preds[f"{model}_seen{suffix}"]
            elif task == "rc":
                method_hits_dataset = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds[f"{model}_seen{suffix}"], labels)]).astype(int)
            else:
                method_hits_dataset = (preds[f"{model}_seen{suffix}"] == labels).astype(int)
            for score in ["seen", "funcOut", "classOut"]:
                key = f"{model}_{score}{suffix}"
                example = "example" in key
                print(f"Comparing {key} with baseline zero") if not example else print(f"Comparing {key} with baseline example")
                with open(f"./results/{task}/suite/{key}_hits.json", "r") as file:
                    method_hits_suite = json.load(file)
                baseline_dataset = baseline_hits_dataset_zero if not example else baseline_hits_dataset_example
                baseline_suite = baseline_hits_suite_zero if not example else baseline_hits_suite_example
                sig_test = get_significance_randomized_g(baseline_dataset, baseline_suite,
                                        method_hits_dataset, method_hits_suite, 10000, seed=42, labels=labels, f1=f1)
                pvalues[key] = sig_test
                if sig_test > .05:
                    print("Difference is not significant")
    if task == "hsd":
        with open(f"./results/{task}/{dataset}pvalue.json", "w") as file:
            json.dump(pvalues, file)  
    else:  
        with open(f"./results/{task}/pvalue.json", "w") as file:
            json.dump(pvalues, file)  
    return pvalues

def get_pvalues_avg(pvalues, tasks, preds, models, methods, tests, verbose=False):
    baseline_hits_dataset_by_task_zero = {}
    baseline_hits_dataset_by_task_example = {}
    baseline_hits_suite_by_task_score_zero = {}
    baseline_hits_suite_by_task_score_example = {}
    labels_by_task = {}
    hate_labels = {}
    for task in tasks:
        if task != "rc":
            labels = tests[task]["label"]
            labels = np.array(labels)
        else:
            labels =[x["text"] for x in tests[task]["answers"]]
        labels_by_task[task] = labels
    for model in models:
        for task in tasks:
            labels = labels_by_task[task]
            if task.split('_')[0]  =="hsd":
                baseline_hits_dataset_by_task_zero[task] = preds[task][f"{model}_baseline_zero"]
                baseline_hits_dataset_by_task_example[task] = preds[task][f"{model}_baseline_example"]
                hate_labels[task] = labels
            elif task == "rc":
                baseline_hits_dataset_by_task_zero[task] = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds[task][f"{model}_baseline_zero"], labels)]).astype(int)
                baseline_hits_dataset_by_task_example[task] = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds[task][f"{model}_baseline_example"], labels)]).astype(int)
            else:
                baseline_hits_dataset_by_task_zero[task]= (preds[task][f"{model}_baseline_zero"] == labels).astype(int)
                baseline_hits_dataset_by_task_example[task] = (preds[task][f"{model}_baseline_example"] == labels).astype(int)
            with open(f"./results/{task.split('_')[0]}/suite/{model}_baseline_zero_hits.json", "r") as file:
                baseline_hits_suite_zero = json.load(file)
            with open(f"./results/{task.split('_')[0]}/suite/{model}_baseline_example_hits.json", "r") as file:
                baseline_hits_suite_example = json.load(file)
            baseline_hits_suite_by_task_score_zero[task] = [deepcopy(baseline_hits_suite_zero) for _ in range(3)]
            baseline_hits_suite_by_task_score_example[task] = [deepcopy(baseline_hits_suite_example) for _ in range(3)]
        for method in methods:
            if (model, method) in pvalues.keys():
                break
            suffix = f"_{method}" if len(method) > 0 else method
            key = f"{model}{suffix}"
            example = "example" in key
            print(f"Comparing {key} with baseline zero") if not example else print(f"Comparing {key} with baseline example")
            method_hits_dataset_by_task = {}
            method_hits_suite_by_task_score = {}
            for task in tasks:
                labels = labels_by_task[task]
                if task.split('_')[0]  =="hsd":
                    method_hits_dataset_by_task[task] = preds[task][f"{model}_seen{suffix}"]
                elif task == "rc":
                    method_hits_dataset_by_task[task] = np.array([metric_max_over_ground_truths(exact_match_score, pred["prediction_text"], label) for pred, label in zip(preds[task][f"{model}_seen{suffix}"], labels)]).astype(int)
                else:
                    method_hits_dataset_by_task[task] = (preds[task][f"{model}_seen{suffix}"] == labels).astype(int)
                for score in ["seen", "funcOut", "classOut"]:
                    key = f"{model}_{score}{suffix}"
                    example = "example" in key
                    with open(f"./results/{task.split('_')[0]}/suite/{key}_hits.json", "r") as file:
                        method_hits_suite_by_task_score.setdefault(task, []).append(json.load(file))
            baseline_dataset = baseline_hits_dataset_by_task_zero if not example else baseline_hits_dataset_by_task_example
            baseline_suite = baseline_hits_suite_by_task_score_zero if not example else baseline_hits_suite_by_task_score_example
            sig_test = randomized_test_avg(baseline_dataset, baseline_suite,
                                    method_hits_dataset_by_task, method_hits_suite_by_task_score, 10000, hate_labels, seed=42, verbose=verbose)
            pvalues[(model, method)] = sig_test 
    return pvalues
        


def get_pvalues_dataset_control(preds, task, dataset, test):
    try:
        with open(f"./data/{task}/{dataset}/pvalues_control.json", "r") as file:
            pvalues = json.load(file)
    except FileNotFoundError:
        pvalues = {}
        if task != "rc":
            labels = test["label"]
        else:
            labels =[x["text"] for x in test["answers"]]
        for prompt_type in ["zero", "example"]:
            for model in np.unique([x.split("_")[0] for x in preds.keys()]):
                for method in ["mismatch"]:
                    key = f"{model}_{method}_{prompt_type}"
                    print(f"Comparing {key} with baseline")
                    try:
                        if task != "hsd":
                            sig_test = get_significance(task, preds[f"{model}_baseline_{prompt_type}"], preds[key], labels)
                            print(sig_test)
                            pvalue = sig_test.pvalue
                        else:
                            pvalue = get_significance_hsd(preds[f"{model}_baseline_{prompt_type}"], preds[key], labels, trials=10000)
                            print(pvalue)
                        pvalues[key] = pvalue
                        if pvalue > .05:
                            print("Difference is not significant")
                    except ValueError:
                        print("Identical predictions")
                        pvalues[key] = np.inf
        with open(f"./data/{task}/{dataset}/pvalues_control.json", "w") as file:
            json.dump(pvalues, file)
    return pvalues

def get_pvalues_suite_control(task, models):
    try:
        with open(f"./data/{task}/suite/pvalues_control.json", "r") as file:
            pvalues = json.load(file)
    except FileNotFoundError:
        pvalues = {}
        for prompt_type in ["zero", "example"]:
            for model in models:
                if task != "hsd" or prompt_type == "zero":
                    with open(f"./results/{task}/suite/{model}_baseline_{prompt_type}_hits.json", "r") as file:
                            hits_baseline = json.load(file)
                for method in ["match", "mismatch", "mixed"]:
                    key = f"{model}_{method}_{prompt_type}"
                    if task != "hsd" or prompt_type == "zero":
                        print(f"Comparing {key} with baseline")
                        with open(f"./results/{task}/suite/{key}_hits.json", "r") as file:
                            hits = json.load(file)                    
                        pvalue = get_significance_randomized(hits_baseline, hits ,trials=10000)
                        print(pvalue)
                        pvalues[key] = pvalue
                        if pvalue > .05:
                            print("Difference is not significant")
                    else:
                        for dataset in ["davidson2017", "founta2018"]:
                            with open(f"./results/{task}/suite/{model}_baseline_{prompt_type}_{dataset}_hits.json", "r") as file:
                                hits_baseline = json.load(file)
                            print(f"Comparing {key} with baseline using {dataset} examples")
                            with open(f"./results/{task}/suite/{key}_hits.json", "r") as file:
                                hits = json.load(file)                    
                            pvalue = get_significance_randomized(hits_baseline, hits ,trials=10000)
                            print(pvalue)
                            pvalues["_".join([key, dataset])] = pvalue
                            if pvalue > .05:
                                print("Difference is not significant")
        with open(f"./data/{task}/suite/pvalues_control.json", "w") as file:
            json.dump(pvalues, file)
    return pvalues

def process_df(df):
    df['model'] = [x.split("_")[0] for x in df.index]
    df['method'] = [x.split("_")[1] for x in df.index]
    df["score"] = df["method"]
    df['method'] = ["Task" if "baseline" in x else f"Task+Rules" for x in df.method]
    df['method'] = [y+"(chatGPT)"  if "from_chatGPT" in x else y for x,y in zip(df.index, df.method)]
    df['method'] =  [y+"+Ex"  if "example" in x else y for x,y in zip(df.index, df.method)]
    df['method'] =  [y+"+Rat"  if "rules" in x else y for x,y in zip(df.index, df.method)]
    df["model"] = df.model.str.split("-").str[-1]
    df = df.sort_values(["model", "method"], key=lambda x: x.map(order))
    return df

def dataset_hits_df(task, result_path, dataset_test, metric, label_col="label"):
    results = load_results(result_path)
    results = {k: v for k, v in results.items() if "seen" in k or "baseline" in k}
    dataset_scores, preds = get_dataset_scores(task, results, dataset_test[label_col], metric)
    labels = dataset_test[label_col] if task != "rc" else [x["text"] for x in dataset_test["answers"]]
    hits = {}
    for model, pred in preds.items():
        if task != "rc":
            hits[model] = (np.array(pred) == np.array(labels)).astype(int)
        else:
            hits[model] = np.array([metric_max_over_ground_truths(exact_match_score, p["prediction_text"], label) for p, label in zip(pred, labels)]).astype(int)
    df = pd.DataFrame(hits).T
    df = process_df(df)
    return df, results


def suite_hits_df(task, path, suite_test):
    hits = load_hits(path)
    all_hits = {}
    for model, hit in hits.items():
        all_hits.setdefault(model, []).extend([x for v in hit.values() for x in v])
    if task == "sa":
        suite_test = suite_test.filter(lambda x: x["functionality"] not in  ['"used to" should reduce', 'reducers'])
    test_id_to_row = collections.OrderedDict()
    if task != "hsd":
        for i, ex in enumerate(suite_test):
            test_id_to_row.setdefault((ex["functionality"], ex["test_id"]), []).append(i)
    df = pd.DataFrame(all_hits).T
    df = process_df(df)
    df = df[(df.score == "seen") | (df.score == "baseline")]
    return df, test_id_to_row
