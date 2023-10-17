from copy import deepcopy

task_configs_dataset = {
    "sa": {
        "name": "Sentiment analysis",
        "format": {
            "mft": "sentence-label pairs",
            "dir": "sentence pairs",
            "inv": "sentence pairs",
        },
        "label": "labels",
        "target": "sentiment",
        "x_col": "sentence",
        "options": "OPTIONS:\n- negative\n- positive",
        "preamble": "In this task, you are given a sentence. You must output the sentence sentiment. Follow these rules:\n",
        "zero": {
            "task_prompt": "Is the sentiment of the following sentence positive or negative (see options at the end)?"
        },
        "example": {
            "task_prompt": "Is the sentiment of the following sentence positive or negative?"
        },
        "label_mapping": ["negative", "positive"],
        "dir_mapping": {
            'not_less_0': "increase negative sentiment",
            'not_less_2': "increase positive sentiment",
            'not_less_conf': "increase prediction confidence",
            'not_more_conf': "decrease prediction confidence"
        },
        "inputs_prefix": "Question:\n",
        "targets_prefix": "Answer:\n",
        "x_y_delimiter":"\n",
        "example_separator": "\n\n",
        "final_suffix": ""
    },
    "pi": {
        "name": "Paraphrase identification",
        "format": {
            "mft": "examples, each containing a pair of questions and a label indicating if they have the same meaning (\"yes\") or not (\"no\")",
            "dir": "examples, each containing two pairs of questions",
            "inv": "examples, each containing two pairs of questions",
        },
        "label": "labels",
        "target": "question similarity",
        "x_col": None,
        "options": "OPTIONS:\n- no\n- yes",
        "preamble": "In this task, you are given two questions. You must indicate if the questions have the same meaning. Follow these rules:\n",
        "zero": {
            "task_prompt": "Do those questions have the same meaning?"
        },
        "example": {
            "task_prompt": "Are these two questions asking the same thing?"
        },
        "label_mapping":  ["no", "yes"],
        "inputs_prefix": "QUES:\n",
        "targets_prefix": "ANS:\n",
        "x_y_delimiter": "\n\n",
        "example_separator": "\n\n\n",
        "final_suffix": ""
    },
    "rc": {
        "name": "Reading comprehension",
        "format": {
            "mft": "examples, each containing a context paragraph, a question about it, and the correct answer",
            "inv": "examples, each containing two context-question pairs",
        },
        "label": "answers",
        "target": "answer",
        "x_col": None,
        "options": None,
        "preamble":  "In this task, you are given a wikipedia article and a question about it. You must extract the answer to the question from the article. Follow these rules:\n",
        "zero": {
            "task_prompt": "Answer a question about this article:"
        },
        "example": {
            "task_prompt": "Answer a question about this article:"
        },
        "label_mapping":  None,
        "inputs_prefix": "The problem: ",
        "targets_prefix": "The answer: ",
        "x_y_delimiter": "\n****\n",
        "example_separator": "\n\n\n",
        "final_suffix": ""
    },
    "hsd": {
        "name": "Hate speech detection",
        "format": {
            "mft": "sentences and labels indicating if a sentence contains hate speech (\"yes\") or not (\"no\")",
        },
        "label": "labels",
        "x_col": "text",
        "options": "OPTIONS:\n- no\n- yes",
        "preamble": "In this task, you are given a sentence. You must indicate if it contains hateful language. Follow these rules:\n",
        "zero": {
            "task_prompt": "Does the following sentence contain hateful language (see options at the end)?"
        },
        "example": {
            "task_prompt": "Does the following sentence contain hateful language?"
        },
        "label_mapping": ["no", "yes"],
        "inputs_prefix": "Question:\n",
        "targets_prefix": "Answer:\n",
        "x_y_delimiter":"\n",
        "example_separator": "\n\n",
        "final_suffix": ""
    },
}

task_configs_suite = deepcopy(task_configs_dataset)
task_configs_suite["sa"]["x_col"] = "test_case"
task_configs_suite["sa"]["options"] = "OPTIONS:\n- negative\n- positive\n- neutral"
task_configs_suite["sa"]["zero"]["task_prompt"] = "Is the sentiment of the following sentence positive, negative or neutral (see options at the end)?"
task_configs_suite["sa"]["example"]["task_prompt"] = "Is the sentiment of the following sentence positive, negative or neutral?"