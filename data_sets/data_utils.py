from datasets import load_from_disk, Dataset
from sklearn.model_selection import train_test_split
from datasets import DatasetDict
import config
import pickle

def get_suite(path, dev=False, hateCheck=False):
    if hateCheck:
        train = path/"hatecheck_train.csv"
        val = path/"hatecheck_val.csv"
        test = path/"hatecheck_test.csv"
        datasets = [Dataset.from_csv(data_path.as_posix(), index_col=0) for data_path in [train, val, test]]
        return combine_datasets(datasets, dev=dev)
    test_dataset = load_from_disk(path)
    if dev:
        test_dataset = test_dataset.filter(lambda x: x["test_id"] in range(1))
    return test_dataset

def load_hsd_dataset(dataset, dev=False):
    with open(config.hsd_data, "rb") as file:
        datasets = pickle.load(file)

    df_raw = datasets[dataset].copy() 

    df_train, df_valtest = train_test_split(df_raw, test_size=0.2, stratify=df_raw.label, random_state=123)
    df_val, df_test = train_test_split(df_valtest, test_size=0.5, stratify=df_valtest.label, random_state=123)
    train = Dataset.from_pandas(df_train, preserve_index=False)
    val = Dataset.from_pandas(df_val, preserve_index=False)
    test = Dataset.from_pandas(df_test, preserve_index=False)
    return combine_datasets([train, val, test])


def combine_datasets(datasets, dev=False):
    return DatasetDict({k: dataset.select(list(range(50))) if dev else dataset for dataset, k in zip(datasets, ["train", "validation", "test"])})

def extract_perturb(dataset):
    base_cases_idxs = []
    perturb_cases_idxs = []
    current_func = None
    current_id = None
    i = 0
    for row in dataset:
        func, test_id = row["functionality"], row["test_id"]
        if row["type"] == "mft":
            base_cases_idxs.append(i)
            i += 1
            continue
        if func == current_func and test_id == current_id:
            perturb_cases_idxs.append(i)
        else:
            base_cases_idxs.append(i)
        current_func = func
        current_id = test_id
        i += 1
    clean_dataset = dataset.select(base_cases_idxs)
    perturbs = dataset.select(perturb_cases_idxs)
    return clean_dataset, perturbs

def get_id_to_samples(dataset):
    dic = {}
    for test_case in dataset:
        test_id, func = test_case["test_id"], test_case["functionality"]
        dic.setdefault(func, {}).setdefault(test_id, []).append(test_case)
    return dic