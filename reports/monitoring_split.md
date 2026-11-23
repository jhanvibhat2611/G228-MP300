# MP300 later-data split

`later.csv` has 300 rows. Sorted by `date`, with `record_id` as the stable
tie-breaker for rows on the same date, the split is:

- **v2 training:** first 210 rows (70%), through `2026-08-10`, record `R20210`.
- **Shared evaluation:** final 90 rows (30%), starting `2026-08-10`, record
  `R20211`, through `2026-08-31`, record `R20300`.

Both sides include August 10 because the 210th and 211th rows share that date.
The record ID boundary makes the split unambiguous. The script
`src/mp300/monitor.py` reproduces this boundary and calculates the monitoring
values recorded in `reports/findings.yaml`.

## Monitoring results

The 10-bin PSI values (cleaned training data as the reference) were:

| Feature | PSI |
| --- | ---: |
| age | 0.0318399819 |
| screen_hours | 0.0235447847 |
| apps_installed | 0.0291133091 |
| plan_gb_limit | 2.4249936091 |
| video_minutes | 0.0329912455 |
| wifi_hours | 0.0238390079 |

`plan_gb_limit` moved the most. The v1 MAEs over all of `later.csv` were
Urban 0.619473 GB, Semi-urban 0.599534 GB, and Rural 2.081179 GB; Rural was
served worst.

On the shared last 90 rows, v1 MAE was 1.058176 GB and v2 MAE was 0.461231 GB.
V2 is 56.4% lower than v1, so it passes the rule requiring at least a 12%
reduction. The champion alias points to model version 2 in the local MLflow
database.

The repository and the latest remote branch did not contain Hardik's model
files or an MLflow database when this work began. These model-dependent results
therefore come from the reproducible local models in
`src/mp300/train_fallback.py`; they are not claimed to be Hardik's earlier
artifacts. Replace them if Hardik provides his own model versions before the
team submits.

The holdout output contains 250 predictions in the input ID order, with only
`record_id,prediction` columns. The script checks the row count, headers, ID
order, and that every prediction is finite. The holdout file has no target, so
its MAE cannot be checked locally.
