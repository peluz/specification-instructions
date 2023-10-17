from pathlib import Path
from utils.results import *
from data_sets.data_utils import load_hsd_dataset
from datasets import load_dataset, load_metric
import pickle

tasks = ["sa", "pi", "rc", "hsd"]
models = ["flan-t5-small", "flan-t5-base", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl", "chatGPT"]
methods = ["", "example", "from_chatGPT_example", "with_rules", "example_with_rules"]
scores =  ["seen", "funcOut", "classOut", "baseline"]
preds_by_task = {}
datasets_by_task = {}
for task in tasks:
    if task == "sa":
        dataset = "sst2"
        dataset_test = load_dataset("glue", "sst2")["validation"]
        result_path = Path(f"./results/sa/sst2/")
        metric = load_metric("glue","sst2")
        label_col = "label"
    elif task == "pi":
        dataset = "qqp"
        dataset_test = load_dataset("glue", "qqp")["validation"]
        metric = load_metric("glue","qqp")
        result_path = Path(f"./results/pi/qqp/")
        label_col = "label"
    elif task == "rc":
        dataset = "squad"
        dataset_test = load_dataset("squad")["validation"]
        result_path = Path(f"./results/rc/squad/")
        metric = load_metric("squad")
        label_col = "answers"
    elif task == "hsd":
        davidson_test = load_hsd_dataset("davidson2017")["test"]
        founta_test = load_hsd_dataset("founta2018")["test"]
        davidson_path = Path(f"./results/hsd/davidson2017/")
        founta_path = Path(f"./results/hsd/founta2018/")
        metric = load_metric("glue","qqp")
        label_col = "label"
    if task != "hsd":
        results = load_results(result_path)
        results = {k: v for k, v in results.items() if any(score in k for score in scores)}
        _, preds = get_dataset_scores(task, results, dataset_test[label_col], metric)
        preds_by_task[task] = preds
        datasets_by_task[task] = dataset_test
        # get_pvalues_g(task, preds, models, methods, dataset, dataset_test)
    else:
        results = load_results(davidson_path)
        results = {k: v for k, v in results.items() if any(score in k for score in scores)}
        _, preds = get_dataset_scores("hsd", results, davidson_test[label_col], metric)
        preds_by_task["hsd_d"] = preds
        datasets_by_task["hsd_d"] = davidson_test
        # get_pvalues_g(task, preds, models, methods, "davidson2017", davidson_test)
        results = load_results(founta_path)
        results = {k: v for k, v in results.items() if any(score in k for score in scores)}
        _, preds = get_dataset_scores("hsd", results, founta_test[label_col], metric)
        preds_by_task["hsd_f"] = preds
        datasets_by_task["hsd_f"] = founta_test
        # get_pvalues_g(task, preds, models, methods,"founta2018", founta_test)

pvalues = get_pvalues_avg(["sa", "pi", "rc", "hsd_d", "hsd_f"], preds_by_task, models, methods, datasets_by_task, verbose=False)
with open(f"./results/pvalues.pickle", "wb") as file:
    pickle.dump(pvalues, file)  

