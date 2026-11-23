# G228-MP300

# G228 · MP300 — Mobile Data Use (Dehradun)

**Course:** SCSE3040 MLOps  
**Batch:** 30  
**Faculty:** Dr. Upasna Joshi

## Project overview

This project predicts a customer's daily mobile data usage (`gb_per_day`) in Dehradun. It is a regression task evaluated using Mean Absolute Error (MAE), measured in GB.

## Team members

| Name | Enrolment number |
| --- | --- |
| Jhanvi Bhat | S24CSEU0687 |
| Ansh Mishra | S24CSEU0635 |
| Hardik Gupta | S24CSEU0751 |
| Kunal Sharma | S24CSEU0655 |
| Junaid Khan | S24CSEU0686 |

## Stage 1 — Data versioning and audit

DVC tracks `train.csv` and `later.csv` through their `.dvc` files in `data/raw/`. The [audit script](src/mp300/audit.py) checks the training data for the seven issue types listed in the assignment. The results are recorded in [reports/findings.yaml](reports/findings.yaml).

## Run the audit locally

Copy `train.csv`, `later.csv`, and `holdout_features.csv` from the original group ZIP into `data/raw/`. Keep the original CSVs intact and store them locally.

From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install dvc pandas
python src/mp300/audit.py
```

## Run and check the prediction API

With the project dependencies installed, put the trained pipeline at
`models/model.joblib` and run the following from the repository root.

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the endpoints. `GET /health` returns
HTTP 200 with `{"status":"ok"}`. Use the following body for `POST /predict`.

```json
{
  "age": 24,
  "screen_hours": 2.5,
  "apps_installed": 32,
  "plan_gb_limit": 8,
  "video_minutes": 0,
  "wifi_hours": 3,
  "phone_tier": "mid",
  "circle": "Semi-urban"
}
```

Junaid's saved pipeline returned HTTP 200 with
`{"prediction":6.263399930617355}` for this request. The prediction is in GB per
day and depends on the deployed model. Changing `plan_gb_limit` to `8.01` or
`video_minutes` to `-1` returns HTTP 422 and identifies the invalid field. A
missing model returns HTTP 503 from `/predict`.

FastAPI validates all eight inputs before inference. The API arranges them in
Jhanvi's `FEATURE_COLUMNS` order in a one-row pandas DataFrame, then passes the
row to the saved pipeline's `predict` method. Joblib loads the pipeline on the
first valid prediction and the process reuses it. `clean_train` is reserved for
raw training data. Successful predictions log the event, status and latency as
JSON to stdout, with customer inputs excluded.

Run `python -m pytest -q` for the test suite. The API tests use a temporary
joblib artifact and run without the raw CSVs.
