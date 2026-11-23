"""Read-only integration verification; run with --zip pointing to the original ZIP."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from pandas.testing import assert_frame_equal
import yaml

from mp300.clean import FEATURE_COLUMNS, clean_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(Path("data/raw").glob("*.csv"))
    before = {p: p.read_bytes() for p in paths}
    with ZipFile(args.zip) as archive:
        for path, content in before.items():
            assert content == archive.read(f"G228_MP300/data/{path.name}"), path
            print(path, "matches ZIP; sha256:", hashlib.sha256(content).hexdigest())

    raw = pd.read_csv("data/raw/train.csv")
    original = raw.copy(deep=True)
    unique = raw.drop_duplicates()
    cleaned = clean_train(raw)
    assert_frame_equal(raw, original)
    assert len(cleaned) == 1000
    assert cleaned.index.equals(unique.index)
    assert cleaned.record_id.equals(unique.record_id)
    assert set(cleaned.phone_tier) == {"basic", "mid", "flagship"}
    assert "gb_from_monthly_bill" not in cleaned
    assert cleaned[FEATURE_COLUMNS].shape == (1000, 8)
    assert cleaned[FEATURE_COLUMNS].notna().all().all()
    semi = unique.circle.eq("Semi-urban")
    pd.testing.assert_series_equal(cleaned.loc[semi, "screen_hours"], unique.loc[semi, "screen_hours"] / 60)
    pd.testing.assert_series_equal(cleaned.loc[~semi, "screen_hours"], unique.loc[~semi, "screen_hours"])
    typos = unique.gb_per_day.gt(10)
    pd.testing.assert_series_equal(cleaned.loc[typos, "gb_per_day"], unique.loc[typos, "gb_per_day"] / 10)
    pd.testing.assert_series_equal(cleaned.loc[~typos, "gb_per_day"], unique.loc[~typos, "gb_per_day"])
    unchanged = ["record_id", "date", "age", "apps_installed", "plan_gb_limit", "video_minutes", "wifi_hours", "circle"]
    assert_frame_equal(cleaned[unchanged], unique[unchanged])
    print("Rows:", len(raw), "->", len(cleaned))
    print("Removed duplicates:", len(raw) - len(unique))
    print("Converted Semi-urban screen times:", int(semi.sum()))
    print("Normalized phone tiers:", int(unique.phone_tier.ne(cleaned.phone_tier).sum()))
    print("Corrected targets:", int(typos.sum()))
    print("Clean target range:", cleaned.gb_per_day.min(), cleaned.gb_per_day.max())

    findings = yaml.safe_load(Path("reports/findings.yaml").read_text())
    expected = {"leaking_column", "unit_mixup", "sentinel_value", "duplicate_rows", "stuck_sensor", "target_typos", "spelling_variants"}
    assert set(findings["data_issues"]) == expected
    for issue in findings["data_issues"].values():
        assert isinstance(issue["present"], bool)
    for path, content in before.items():
        assert path.read_bytes() == content, f"Raw file changed: {path}"
    print("PASS: cleaning, seven YAML sections, and raw-file integrity")


if __name__ == "__main__":
    main()
