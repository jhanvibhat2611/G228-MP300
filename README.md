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
