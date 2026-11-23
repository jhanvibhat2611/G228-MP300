# MP300 Stage 4 monitoring and predictions

## Reproduce the work

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m src.mp300.monitor
.\.venv\Scripts\python.exe -m pytest tests\test_monitoring.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

The monitoring tests use synthetic frames and a temporary SQLite registry. They
do not need the project CSVs. The real-data command reads `train.csv` and calls
`clean_train` once. It never applies that cleaner to `later.csv` or
`holdout_features.csv` because their screen-time values are already in hours.

## PSI method and results

The supplied group ZIP and repository did not include the course P13 PSI
instructions. This run uses the following explicit method: make 10 equally
spaced quantile cuts from cleaned training data, collapse repeated cuts, and
add `-infinity` / `+infinity` outer edges so every later value is counted. Each
bin follows NumPy histogram boundaries (left-closed, right-open; the last bin
includes its right edge). Add 0.5 to each reference and later bin count before
normalizing, including empty bins. PSI is the sum of
`(later_share - train_share) * ln(later_share / train_share)`.

All six features had 10 distinct effective bins for these data. The most
drifted feature was `plan_gb_limit`.

| Feature | PSI |
| --- | ---: |
| age | 0.02014478096060273 |
| screen_hours | 0.028974002024107843 |
| apps_installed | 0.030974680627314188 |
| plan_gb_limit | 2.2594929552838026 |
| video_minutes | 0.028683420540921684 |
| wifi_hours | 0.028086037826865275 |

## Group MAE and retraining split

V1 was loaded specifically as registered model version 1. Its MAE on all 300
later rows was:

| Circle | V1 MAE (GB) |
| --- | ---: |
| Urban | 0.34590717431835977 |
| Semi-urban | 0.3133281695440838 |
| Rural | 1.7645662187227251 |

Rural was served worst. The split matches Hardik's
`chronological_split`: sort by `date`, `record_id`, then original row order;
take rows 1–210 for the retraining portion and rows 211–300 for the shared
comparison. The boundary is `2026-08-10 / R20210` then
`2026-08-10 / R20211`.

The ordered record-ID fingerprints use SHA-256 over UTF-8 IDs joined with LF,
including a final LF:

- First 210 IDs: `092780735f54622b052c8f9ed6b5d35e1ab8335814526fdbb3651e9bc000b677`
- Final 90 IDs: `c1517965f44d69a958d3de829877c23e5b788c7a9ab3104be8cca411085bb303`

On the identical final 90 rows, v1 MAE was `0.7742215992018804` GB and v2 MAE
was `0.5727945150379006` GB. V2 reduced MAE by 26.01672239209347%, which passes the
required 12% rule. The registry already points `champion` to version 2; the
monitoring run checked that alias and did not change it. The v1 and v2 run IDs
were `bc82d6694d194f30b788bdcfc2ed5406` and
`5c310488625847ab8416a2fb8733c013`.

## Champion and holdout

The registered `champion` model was resolved from the SQLite registry and
compared against `models/model.joblib`. Their holdout predictions matched
exactly (maximum absolute difference `0`). The output
`predictions/holdout_predictions.csv` has 250 rows, the two required columns
`record_id,prediction`, the original holdout ID order, and finite numeric
predictions.

All original CSVs matched the files in `G228_MP300.zip` byte-for-byte both
before and after analysis:

- `train.csv`: `4534b73fe3274c963a6ff3034e58b0a0483479f23996ffd74e2a8df383ecef87`
- `later.csv`: `809cb69a3a471af84025a7607de735a20223b005f27ed4d269e16a896094b955`
- `holdout_features.csv`: `e7da694eedabdae203c30c31d204b9bdfbf2e9cd97125a7a7577eb908b305765`

No monitoring dependency was added: the code uses the already pinned
`joblib`, `numpy`, `pandas`, and `PyYAML`, plus Python's built-in SQLite
reader. It reads the registered model paths and champion alias without
changing `mlflow.db`.

## Handoff for Hardik

Prepared for handoff; it has not been sent. The actual registered v1 and v2
artifacts were loaded and independently scored on the shared last 90 rows:
`0.7742215992018804` GB and `0.5727945150379006` GB. V2 passes the 12%
promotion rule, and the existing champion alias is version 2. The split
fingerprints above let the training and monitoring rows be checked exactly.
