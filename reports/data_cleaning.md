# G228 / MP300 data cleaning review

Read the original `PROBLEM.md` and `STUDENT_HANDBOOK.md` directly from
`/Users/jhanvibhat/Downloads/G228_MP300.zip` before implementation. All three
local CSVs were compared byte-for-byte against that archive, before and after
cleaning. No CSV was written. No commit, staging, or push was performed.

## Seven audit checks on original train.csv

The existing audit and independent pandas calculations were run before editing
findings. Counts here use all 1,025 original rows unless stated otherwise.

| Check | Result and evidence |
| --- | --- |
| Leaking column | `gb_from_monthly_bill` is absent from holdout and is available only after the event, according to PROBLEM.md. Correlation is 0.294859 with the corrupt target and 0.998438 after correcting target typos. Timing establishes leakage; correlation supports it. |
| Unit mixup | Semi-urban screen time is 30–714, mean 376.51. Dividing by exactly 60 gives 0.5–11.9 hours, mean 6.275. Rural and Urban means are 5.92 and 6.28 hours. Clean later.csv also uses hours. |
| Sentinel | Absent. None of -1, -999, 999, 9999 occurs in any numeric column. Numeric ranges and most frequent values were also inspected, not just that shortlist. No missing values occur. Zero video/Wi-Fi readings are valid; elevated screen times and targets are explained by the separate issues. No evidence supports sentinel replacement. |
| Duplicates | 25 exact extra copies; removing them leaves 1,000 rows. |
| Stuck sensor | Absent. For every numeric column, the longest consecutive calendar-day run with the same value on every row is only one day. January 31 has one row, so a constant value that day is not evidence of a stuck device. All 181 calendar days are represented. An independent date loop confirmed the grouped audit result. |
| Target typos | 13 values occupy 26.51–46.81 GB. All other targets are at most 6.738. The largest adjacent sorted gap is 19.772 GB, from 6.738 to 26.51. The fixed rule `gb_per_day > 10`, divided by ten, maps those 13 values to 2.651–4.681. This is a rule for this original training set, not a universal usage limit. |
| Spelling variants | 18 raw spellings collapse to three tiers: basic, mid, flagship. Strip whitespace, lowercase, then map bas./mid./fla. Clean later.csv confirms the three canonical levels. |

The intended sentinel/stuck conclusions remain false. The actual findings
file had five section headings indented under other issues, with repeated
`present` and `column` keys that a YAML loader could overwrite. Those headings
are now siblings under `data_issues`; all seven sections parse correctly.
Tracking and monitoring fields remain for their owners to fill.

## Cleaning result

Cleaning deduplicates first, so correction counts below exclude extra copies.

| Operation | Count |
| --- | ---: |
| Input rows | 1,025 |
| Removed exact duplicate copies | 25 |
| Output rows | 1,000 |
| Semi-urban screen values divided by 60 | 532 |
| Phone-tier strings changed | 150 |
| Targets divided by ten | 13 |
| Billing columns removed | 1 |

Before deduplication, there are 545 Semi-urban rows and 155 noncanonical tier
strings. The cleaned target range is 0.473–6.738 GB. Dates, IDs, row order,
original retained index labels, and unrelated feature values are preserved.
The result has 11 columns when given the original 12-column training data.

## Interface for the model lead

The model lead can use this handoff. Use this handoff in your
training code when you are ready; no training/API/deployment code was edited:

```python
import pandas as pd
from mp300.clean import FEATURE_COLUMNS, TARGET_COLUMN, clean_train

raw = pd.read_csv("data/raw/train.csv")
cleaned = clean_train(raw)
X = cleaned.loc[:, FEATURE_COLUMNS]
y = cleaned.loc[:, TARGET_COLUMN]
```

`FEATURE_COLUMNS` contains exactly the eight request fields from PROBLEM.md,
in its documented order. It excludes the target, record_id, date and bill.
The cleaner also drops the bill physically, so it cannot accidentally be
selected from the returned frame. It keeps the target and metadata for
splitting and evaluation. It raises a clear ValueError on missing required
columns or unrecognized/missing phone tiers, rather than inventing categories.

Call `clean_train` once on raw train.csv, before train/validation splitting,
so exact copies cannot leak across the split. It is intentionally not
idempotent: a second call would divide Semi-urban hours by 60 again. Never run
it on later.csv, holdout or API requests; those already have correct units.
For those inputs, select `FEATURE_COLUMNS` directly and use the model lead's
fitted preprocessing pipeline. Later data can legitimately exceed 10 GB
(the observed maximum is 10.203), so do not reuse the training typo rule there.
Do not add this cleaner as a transformer that runs at prediction time.

## Files changed and why

- `src/mp300/clean.py`: pure cleaning function, named feature list and documented training-only cutoff. It returns a copy and performs no disk I/O.
- `src/mp300/audit.py`: retains the seven checks; adds explicit largest-gap and billing-correlation evidence, missing-value counts, and calendar-aware sensor runs. Missing readings break runs. A main guard avoids reading files when importing the module. Much of the diff is indentation into main().
- `reports/findings.yaml`: repairs nesting of five issue headings and adds evidence comments for the two absent issues. No tracking or monitoring results were invented.
- `tests/test_clean.py`: five synthetic unittest tests, also discoverable by pytest. They check nonmutation, all observed spelling forms, exact duplicate semantics, conversions, leakage exclusion, cutoff behavior and useful invalid-input errors. No raw files are required for these tests.
- `src/mp300/verify_clean.py`: read-only integration check for real-data corrections, unchanged unrelated values, raw-file equality to ZIP and seven parsed YAML sections. It requires local CSVs and the original ZIP, so it is separate from CI tests.
- `reports/data_cleaning.md`: this evidence, explanation and model handoff.

## Exact verification commands run

From `/Users/jhanvibhat/G228-MP300`:

```sh
.venv/bin/python src/mp300/audit.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m mp300.verify_clean --zip /Users/jhanvibhat/Downloads/G228_MP300.zip
git diff --check
git diff -- reports/findings.yaml
git status --short
```

The audit was also captured with:

```sh
.venv/bin/python src/mp300/audit.py > /tmp/mp300-audit.txt
tail -18 /tmp/mp300-audit.txt
```

All five tests passed. Real-data integration checks passed. `git diff --check`
passed. Initial inspection also used read-only inline Python to read ZIP
instructions, compare bytes, inspect summaries, independently calculate
correction counts/gaps/correlation and scan constant-day runs. The durable
commands above reproduce the relevant evidence from src/.

Environment tested: Python 3.12.10, pandas 3.1.1, existing PyYAML installation.
Pytest is not installed locally, so tests were run with the standard-library
unittest runner. No dependencies were installed or teammates' dependency
configuration changed. The future shared pinned environment/CI still needs
its own test run; no model-quality or end-to-end service claim is made here.

Original train SHA-256:
`4534b73fe3274c963a6ff3034e58b0a0483479f23996ffd74e2a8df383ecef87`.
The verification command prints SHA-256 for all three files.

No unresolved data-cleaning questions. Model integration is handed off to you;
training, API and deployment remain outside this change. Pre-existing
untracked .DS_Store files and holdout_features.csv were left untouched.
