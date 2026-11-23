"""Small synthetic tests: no raw CSVs needed in CI; also discoverable by pytest."""
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from mp300.clean import FEATURE_COLUMNS, clean_train


def sample():
    return pd.DataFrame({
        "record_id": ["A", "B", "C"], "date": ["2026-01-01"] * 3,
        "age": [20, 30, 40], "screen_hours": [90, 6, 5],
        "apps_installed": [20, 40, 60], "plan_gb_limit": [2, 3, 4],
        "video_minutes": [0, 60, 120], "wifi_hours": [0, 4, 8],
        "phone_tier": [" BAS. ", "mid.", "Fla."],
        "circle": ["Semi-urban", "Urban", "Rural"],
        "gb_from_monthly_bill": [2.65, 6.74, 1.2],
        "gb_per_day": [26.51, 6.738, 1.2],
    })


class CleanTrainTests(unittest.TestCase):
    def test_corrections_and_input_unchanged(self):
        raw = sample()
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        before = raw.copy(deep=True)
        result = clean_train(raw)
        assert_frame_equal(raw, before)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.index.tolist(), [0, 1, 2])
        self.assertEqual(result.screen_hours.tolist(), [1.5, 6, 5])
        self.assertEqual(result.phone_tier.tolist(), ["basic", "mid", "flagship"])
        self.assertAlmostEqual(result.gb_per_day.iloc[0], 2.651)
        self.assertEqual(result.gb_per_day.iloc[1:].tolist(), [6.738, 1.2])
        self.assertNotIn("gb_from_monthly_bill", result.columns)
        self.assertEqual(result[FEATURE_COLUMNS].shape, (3, 8))
        result.loc[0, "age"] = 99
        assert_frame_equal(raw, before)

    def test_only_exact_duplicates_removed_before_leak_is_dropped(self):
        raw = pd.concat([sample().iloc[[0]]] * 2, ignore_index=True)
        raw.loc[1, "gb_from_monthly_bill"] = 99
        self.assertEqual(len(clean_train(raw)), 2)

    def test_all_observed_spelling_forms(self):
        for tier, abbrev in [("basic", "bas."), ("mid", "mid."), ("flagship", "fla.")]:
            for spelling in [tier, tier.upper(), tier.title(), " " + tier, tier + " ", abbrev]:
                with self.subTest(spelling=spelling):
                    raw = sample()
                    raw["phone_tier"] = spelling
                    self.assertEqual(clean_train(raw).phone_tier.tolist(), [tier] * 3)

    def test_invalid_inputs_fail_without_mutation(self):
        for value in ["premium", None]:
            raw = sample()
            raw.loc[0, "phone_tier"] = value
            before = raw.copy(deep=True)
            with self.assertRaisesRegex(ValueError, "phone_tier"):
                clean_train(raw)
            assert_frame_equal(raw, before)
        with self.assertRaisesRegex(ValueError, "gb_per_day"):
            clean_train(sample().drop(columns="gb_per_day"))

    def test_optional_leak_and_metadata_and_cutoff(self):
        raw = sample().drop(columns=["record_id", "date", "gb_from_monthly_bill"])
        raw["gb_per_day"] = [10, 10.01, 0]
        result = clean_train(raw)
        self.assertEqual(result.gb_per_day.iloc[0], 10)
        self.assertAlmostEqual(result.gb_per_day.iloc[1], 1.001)
        self.assertEqual(result.gb_per_day.iloc[2], 0)


if __name__ == "__main__":
    unittest.main()
