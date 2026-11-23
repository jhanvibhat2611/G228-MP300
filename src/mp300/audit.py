import pandas as pd


def main() -> None:
    """Print all seven checks on the untouched CSVs; run from repository root."""
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

    for column in ["phone_tier", "circle"]:
        print(f"\nCategory spellings for {column}:")
        for name, data in [("train.csv", train), ("later.csv", later)]:
            counts = data[column].value_counts()
            print(name, {repr(value): count for value, count in counts.items()})

    print("\nTop 40 gb_per_day values:")
    print(train["gb_per_day"].nlargest(40).to_list())

    print("\nTarget typo rows:", int(train["gb_per_day"].gt(10).sum()))

    print("\nLongest constant stretch per numeric column:")
    for column in train.select_dtypes(include="number").columns:
        daily = train.groupby("date", sort=True)[column].agg(["min", "max", "count", "size"])
        # Missing calendar days break runs; partial/missing readings are not constant.
        daily.index = pd.to_datetime(daily.index)
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
        constant = daily["min"].where(daily["min"].eq(daily["max"]) & daily["count"].eq(daily["size"]))
        run_ids = constant.ne(constant.shift()).cumsum()

        runs = constant.dropna().groupby(run_ids)
        longest = max((run for _, run in runs), key=len, default=None)

        if longest is not None:
            print(column, longest.index[0], longest.index[-1],
                  f"{len(longest)} recorded days", f"value={longest.iloc[0]}")

    print("\nMissing readings per column:")
    print(train.isna().sum().to_dict())

    ordered = train[target].sort_values().reset_index(drop=True)
    gap_index = int(ordered.diff().idxmax())
    print("\nLargest target gap:", ordered.iloc[gap_index - 1], "to", ordered.iloc[gap_index])
    corrected_target = train[target].where(train[target].le(10), train[target] / 10)
    print("Billing correlation with raw target:", train["gb_from_monthly_bill"].corr(train[target]))
    print("Billing correlation with corrected target:", train["gb_from_monthly_bill"].corr(corrected_target))
    print("Billing column is post-event per PROBLEM.md and absent from holdout; exclude it.")


if __name__ == "__main__":
    main()
