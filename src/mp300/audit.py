import pandas as pd

train = pd.read_csv("data/raw/train.csv")
duplicate_count = int(train.duplicated().sum())
print("Extra duplicate rows:", duplicate_count)
