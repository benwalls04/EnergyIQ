# Campus EnergyIQ

Flags unusual hourly energy use for three campus buildings (Ohio Stadium, Thompson Memorial Library, Scott House). The system replays two years of historical meter data in time order, never using data from the future, and marks readings that fall outside what's normal for that meter at that time.

## The approach

The pipeline has three steps. **Steps 1 and 2 are built; step 3 is next.**

1. **Expected usage (model).** One gradient-boosted tree model (`HistGradientBoostingRegressor`) covers every building and utility. It predicts each meter's usage from calendar and ID features only: season, weekday vs. weekend, hour bucket, utility, meter, building, and hour/month as sine/cosine. It doesn't use weather or occupancy. The model gives the dashboard its "expected" line. The goal is a reasonable baseline, **not** the lowest possible error. A model fitted too closely would learn game days and cold snaps as normal, and then couldn't flag them.
2. **Candidate anomalies (thresholds).** Readings are split into **groups**: meter × month × weekday/weekend × hour bucket. Each group gets a fixed lower and upper bound from its training readings. A reading outside its group's bounds is flagged as a *candidate* anomaly.
3. **Events (LLM, not built yet).** An LLM will merge the candidates into real events (e.g. "Scott House heat ran high all weekend") and write the alert summaries.

### How the bounds are calculated

For each group, using only readings from before the training cutoff:

```
median       = median reading
lower_spread = median distance below the median   (readings below it only)
upper_spread = median distance above the median   (readings above it only)

lower_bound  = max(0, median - k * lower_spread)
upper_bound  = median + k * upper_spread
```

- `k` is `ANOMALY_K` in `config.py`.
- The two sides are measured separately because energy spikes are usually bigger than drops. A single symmetric width would hide drops.
- A flagged reading also gets a **severity**: `(reading - median) / (distance from the median to the bound it crossed)`.
  - Above **+1** is a spike, below **−1** is a drop. +3 means three times as far out as the bound.
- A group that never appeared in the training data has no bounds, so its readings are never flagged.

Hour buckets are 4-hour blocks:

| `hour_bucket` | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| hours | 12–3am | 4–7am | 8–11am | 12–3pm | 4–7pm | 8–11pm |

## Setup

You need **Python 3.10+**. Run every command from inside `backend/`.

```bash
cd EnergyIQ/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell (every new terminal)
source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
```

If PowerShell says scripts are disabled, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once. In VS Code, pick `backend/.venv` with **Python: Select Interpreter**.

## Building the database

The database (`data/db/energyiq.duckdb`) isn't committed, so everyone builds their own from `data/raw/`. There are two ways to build it, and both create the **same tables**:

| Command | What you get | Time |
|---|---|---|
| `python -m scripts.build_db` | All readings and weather. **No anomalies:** `is_anomalous` is false everywhere and `group_thresholds` is empty. | ~1 min |
| `python -m scripts.build_db --replay` | Same data, plus flagged readings and per-group thresholds. **Use this for the dashboard and anything anomaly-related.** | several min |

Both drop and rebuild everything, so they're safe to rerun.

### What `--replay` does

It feeds the data in one 6-month segment at a time (`TIME_SEGMENTS` in `config.py`), the way the system would have seen it live:

1. **Starting state.** Loads everything before `TRAIN_END` (`2025-09-01`, i.e. Sep 2023 – Aug 2025). Trains the model, computes thresholds, and saves both. Nothing in this period is flagged; it's the starting history.
2. **For each later segment** (`sept2025-feb2026`, then `march2026-sept2026`):
   1. Inserts the segment's readings and weather.
   2. Flags them using the thresholds trained **before** the segment, which guarantees no lookahead. Flagged rows get `is_anomalous = true` and a `severity`.
   3. Retrains on everything so far, flagged readings included, and writes a new set of thresholds keyed by the segment's end date.

The output looks like this:

```
INITIAL STATE: data before 2025-09-01
-- training on data before 2025-09-01 --
thresholds: <g> groups, smallest n_train ..., ...
wrote <g> group thresholds (train_end 2025-09-01)
BATCH sept2025-feb2026: 2025-09-01 to 2026-03-01
sept2025-feb2026: <n> of <N> readings flagged (x.xx%)
...
DONE: wrote .../data/db/energyiq.duckdb
```

Each training run also saves its model to `energyiq/ml/models/replay/<train_end>.joblib`.

### Gotchas

- **Always use `python -m` with dotted names** (`python -m scripts.build_db`, not `python scripts/build_db.py`). Otherwise you'll get `No module named 'energyiq'`.
- **Close the database in other tools first** (e.g. the VS Code DuckDB extension). Otherwise the build fails with "file is being used by another process".
- **If the build fails while dropping tables** (left over from the old `anomalies` table), delete `data/db/energyiq.duckdb` and rebuild.
- **`--replay` rewrites the committed models** in `energyiq/ml/models/replay/`. If you only wanted the database, discard them with `git checkout -- energyiq/ml/models/replay`.

To browse the database, install the **DuckDB** VS Code extension, then run **DuckDB: Select Database → Attach** on `data/db/energyiq.duckdb`. Detach it before rebuilding.

## Database tables

| Table | Contents |
|---|---|
| `buildings` | `id`, `sitename`, `simscode` |
| `meter_readings` | One row per meter per reading: `meterid`, `readingtime`, `building_id`, `utility` (`ELECTRICITY`, `HEAT`, ...), `readingvalue`, `readingunits`, **`is_anomalous`**, **`severity`** (null unless flagged) |
| `weather` | Hourly campus weather. Loaded, but not used by the model right now |
| `group_thresholds` | One row per group per training run: `train_end`, `meterid`, `month`, `is_weekend`, `hour_bucket`, `n_train`, `k`, `median`, `lower_bound`, `upper_bound` |

The `group_thresholds` rows for `train_end = X` are the bounds used to flag the segment that starts at `X`. The last `train_end` (`2026-09-01`) is the "current" set, trained on all the data.

Useful queries:

```sql
-- how much was flagged, per building and utility
SELECT b.sitename, m.utility, count(*) FILTER (WHERE m.is_anomalous) AS flagged, count(*) AS total
FROM meter_readings m JOIN buildings b ON b.id = m.building_id
WHERE m.readingtime >= '2025-09-01'
GROUP BY ALL ORDER BY ALL;

-- the most extreme candidates
SELECT * FROM meter_readings WHERE is_anomalous ORDER BY abs(severity) DESC LIMIT 20;

-- the bounds a reading was judged against (the dashboard can draw these as a band)
SELECT * FROM group_thresholds
WHERE train_end = '2025-09-01' AND meterid = <id>
ORDER BY month, is_weekend, hour_bucket;
```

## Training experiments

`train.py` is for checking the model on its own. It trains on data before `TRAIN_END`, tests on everything after, and prints the results. It doesn't write to the database. Run a plain or replay build first, since it reads from the database.

```bash
python -m energyiq.ml.train -v       # train, test, print metrics + breakdown
python -m energyiq.ml.train --save   # also save to energyiq/ml/models/experiments/ (not committed)
```

- **Cleaning (training rows only):** short runs of zero readings are filled in, long ones (outages) are dropped, extreme outliers are dropped, and readings are averaged to one per meter per hour. Test rows are left raw.
- **Metrics:** MAE, RMSE and R² on the test period, after dropping the largest 2% of errors (`TRIM_PCT` in `evaluation.py`).
- **`-v` breakdown:** a tree of MAE/RMSE, utility → building → meter → month. Each level is the plain average of the groups below it. Errors are in each meter's own units, so only compare within a utility.

## Where to change things

| Setting | File | What it does |
|---|---|---|
| `ANOMALY_K` | `config.py` | Width of the bounds. Higher means fewer flags |
| `TRAIN_END` | `config.py` | Where the starting history ends and replay/testing begins |
| `TIME_SEGMENTS` | `config.py` | The data folders and their date ranges. To add data, add a folder to `data/raw/` and an entry here |
| `HGB_DEFAULTS` | `config.py` | Model settings |
| `HOUR_BUCKET_HOURS`, `GROUP_COLS`, `FEATURE_COLS` | `ml/preprocess.py` | Hour bucket size, threshold groups, model features |
| `OUTLIER_K`, `ZERO_RUN_MAX_SPAN` | `ml/preprocess.py` | How aggressively training data is cleaned |
| `BREAKS`, `STADIUM_EVENTS` | `config.py` | Academic breaks and stadium events. Not used yet; kept as context for the LLM step |

If you change anything that affects flagging, rerun `--replay` and mention it in your PR, since everyone's anomalies will change.

## Repo layout

```
backend/
├── energyiq/
│   ├── config.py           # settings (see table above)
│   ├── models.py           # database tables
│   ├── ingest.py           # CSV loading, database writes
│   └── ml/
│       ├── preprocess.py   # DB rows -> features, groups, training-data cleaning
│       ├── train.py        # experiment training (train) and replay training (train_replay)
│       ├── thresholds.py   # per-group bounds and flagging
│       ├── evaluation.py   # metrics and the -v breakdown
│       ├── sweep.py        # W&B hyperparameter sweeps (out of date, being reworked)
│       └── models/replay/  # models saved by --replay
├── scripts/build_db.py     # builds the database (plain or --replay)
└── requirements.txt
data/
├── raw/                    # source CSVs: one folder per 6-month segment + weather_data/
└── db/                     # energyiq.duckdb (not committed)
```

## Not done yet

- LLM step: merge candidates into events and write alert summaries (step 3).
- Simulated alert emails.
- Dashboard.
- `sweep.py` still targets the old training setup.

## Adding a dependency

1. `pip install <package>` with the venv active.
2. Add it to `backend/requirements.txt` and commit.
3. Teammates run `pip install -r requirements.txt` after pulling.