from pathlib import Path

openai_key = "YOUR KEY HERE"

sa_path = Path("./data/sa/sa")
pi_path = Path("./data/qqp/qqp")
rc_path = Path("./data/squad/squad")
hatecheck_path = Path("./data/hatecheck/csvs")
hsd_data = Path(".data/hsd/datasets/training_data_binary.pkl")

sa_suite = Path("./data/release_data/sentiment/sentiment_suite.pkl")
pi_suite = Path("./data/release_data/qqp/qqp_suite.pkl")
rc_suite = Path("./data/release_data/squad/squad_suite.pkl")