"""Dataset-specific cleaning for the original MP300 training data."""

import pandas as pd

FEATURE_COLUMNS = [
    "age", "screen_hours", "apps_installed", "plan_gb_limit",
    "video_minutes", "wifi_hours", "phone_tier", "circle",
]
TARGET_COLUMN = "gb_per_day"
# Original train.csv has no targets between 6.738 and 26.51 GB.
# This fixed cutoff isolates its 13 extra-zero typos; it is not a general limit.
TARGET_TYPO_THRESHOLD = 10.0


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of raw MP300 train.csv, retaining index and order.

    Keep the first exact duplicate, convert Semi-urban minutes to hours,
    normalize phone tiers, divide the identified target typos by ten, and
    remove the post-event billing column. Retain record_id/date if supplied.
    Select FEATURE_COLUMNS for X and TARGET_COLUMN for y after cleaning.

    Call ONCE on raw training data, before splitting. Do not call on already
    cleaned data, later.csv, holdout, or API requests: their units are hours.
    Unknown/missing phone tiers raise ValueError rather than silently becoming
    missing categories. This function performs no file reads or writes.
    """
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")

    cleaned = df.drop_duplicates().copy(deep=True)
    tiers = cleaned["phone_tier"].astype("string").str.strip().str.lower()
    tiers = tiers.replace({"bas.": "basic", "mid.": "mid", "fla.": "flagship"})
    invalid = ~tiers.isin(["basic", "mid", "flagship"])
    if invalid.any():
        raise ValueError(f"Unknown or missing phone_tier: {cleaned.loc[invalid, 'phone_tier'].tolist()}")
    cleaned["phone_tier"] = tiers

    # Cast first so integer-only test/input columns can safely hold fractions.
    cleaned["screen_hours"] = cleaned["screen_hours"].astype(float)
    semi_urban = cleaned["circle"].eq("Semi-urban")
    cleaned.loc[semi_urban, "screen_hours"] /= 60.0
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(float)
    typos = cleaned[TARGET_COLUMN].gt(TARGET_TYPO_THRESHOLD)
    cleaned.loc[typos, TARGET_COLUMN] /= 10.0

    return cleaned.drop(columns=["gb_from_monthly_bill"], errors="ignore")
