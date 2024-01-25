Title here
 
This repo holds source code for the paper [Title pending](link here).


## Requirement

- [Anaconda](https://www.anaconda.com/download)

## Setting up 

1. Run the snippet below to install all dependencies:

```console
conda env create -f environment.yml
```
2. Download the test suites from https://github.com/marcotcr/checklist/blob/master/release_data.tar.gz and extract the files to the 'data' folder.
3. Refer to https://github.com/peluz/beluga to convert suites to datasets with predefined splits.
4. Download the hate speech data from https://github.com/paul-rottger/hatecheck-experiments/blob/master/Data/Clean%20Training%20Data/training_data_binary.pkl and fix the hsd_data path in the config file.

## Reproducing the experiments
- Script human_rules.ipynb generates the human-made rules
- Script chatGPT_rules.py generates ChatGPT-made functionality rules
- Script get_predictions.py obtains model predictions. Usage:
    ```console
    python get_predictions.py {task} --model_url {model_url} --methods {methods} {--chatGPT} {--ask_rule} {--add_examples} {--from_chatGPT} --bs {bs}
    ```
    - {task} can be sa, pi, rc or hsd.
    - {model_url} corresponds to the model url in the Hugginface model zoo (e.g., google/flan-t5-XXL)
    - {method} can be seen, funcOut, classOut or all.
    - add --chatGPT if prompting chatGPT
    - add --ask_rule to add the Rational module
    - add --add_examples to add the Exemplars module
    - add --from_chatGPT to use chatGPT-generated rules. Do not add it to use human-generated rules.
    - {bs} sets the number of prompts in a batch for generation.
- To evaluate task performance:
    - Run process_suite_preds.ipynb
    - Run process_suite_results.ipynb
    - Run evaluate.ipynb
- To evaluate model rationales:
    - Run rat_eval.ipynb
- Script 'significance_testing.py' runs the significance tests.

## Model's generations
- All model generations are available in the "results" folder
- All ChatGPT responses are available in the "responses" folder


## Data notice
- Our data folder contains the splits we use for the HateCheck data. The original suite can be found at https://github.com/paul-rottger/hatecheck-data and is distributed under a CC-BY-4.0 license.