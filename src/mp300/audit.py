import pandas as pd

train = pd.read_csv("data/raw/train.csv")
duplicate_count = int(train.duplicated().sum())
print("Extra duplicate rows:", duplicate_count)

holdout = pd.read_csv("data/raw/holdout_features.csv")
target = "gb_per_day"
train_only = sorted(set(train.columns) - set(holdout.columns) - {target})

print("Columns only in training data, excluding the target:", train_only)

print("\nSentinel candidates (value → row count):")
for column in train.select_dtypes(include="number").columns:
    counts = train[column].value_counts()
    matches = counts[counts.index.isin([-1, -999, 999, 9999])]
    if not matches.empty:
        print(column, matches.to_dict())

print("\nNumeric column ranges and most repeated values:")
for column in train.select_dtypes(include="number").columns:
    values = train[column]
    print(f"{column}: min={values.min()}, max={values.max()}")
    print("  Top 5 (value → count):", values.value_counts().head(5).to_dict())

later = pd.read_csv("data/raw/later.csv")

for name, data in [("train.csv", train), ("later.csv", later)]:
    print(f"\nscreen_hours by circle in {name}:")
    print(data.groupby("circle")["screen_hours"].agg(["min", "mean", "max"]).round(2))
