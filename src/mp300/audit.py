import pandas as pd

train = pd.read_csv("data/raw/train.csv")
duplicate_count = int(train.duplicated().sum())
print("Extra duplicate rows:", duplicate_count)

holdout = pd.read_csv("data/raw/holdout_features.csv")
target = "gb_per_day"
train_only = sorted(set(train.columns) - set(holdout.columns) - {target})

print("Columns only in training data, excluding the target:", train_only)
