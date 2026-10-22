# Campus EnergyIQ

Predicts each campus building's expected hourly energy use from weather and calendar data, then flags hours where actual usage is unusually far from that prediction. Buildings: Ohio Stadium, Thompson Memorial Library, Scott House.

## Repo layout

```
EnergyIQ/
├── backend/
│   ├── energyiq/
│   │   ├── config.py        # paths, time segments, default features, HGB settings, ANOMALY_K
│   │   ├── models.py        # database table definitions
│   │   ├── ingest.py        # loads the raw CSVs into the database
│   │   ├── anomaly_detector.py # flags anomalies in a batch of readings using saved replay models
│   │   └── ml/
│   │       ├── preprocess.py  # turns database rows into model features; academic breaks and stadium events live here
│   │       ├── thresholds.py  # per-group anomaly thresholds (shared by training and the detector)
│   │       ├── train.py       # trains and scores one model; also trains the replay models
│   │       ├── evaluation.py  # metrics and helpers shared by train.py/sweep.py
│   │       ├── sweep.py       # runs many training configs, logs them to W&B
│   │       └── models/
│   │           ├── experiments/  # models saved by `train.py --save`, one folder per utility
│   │           └── replay/       # models saved by `build_db --replay`, one folder per training cutoff
│   ├── scripts/
│   │   └── build_db.py      # builds the database from the raw CSVs
│   └── requirements.txt
└── data/
    ├── raw/                 # source CSVs, one folder per 6-month segment (Sep 2023 – Aug 2026) + weather_data/
    └── db/                  # energyiq.duckdb is written here (not committed)
```

## Setup

You need **Python 3.10+**. All commands are run from inside `backend/`.

```bash
cd EnergyIQ/backend
python -m venv .venv
```

Activate the environment (every new terminal):

```bash
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
source .venv/bin/activate        # macOS/Linux
```

If PowerShell says scripts are disabled, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once.

In VS Code, pick `backend/.venv` via **Python: Select Interpreter**.

Install dependencies and build the database:

```bash
pip install -r requirements.txt
python -m scripts.build_db
```

This reads everything in `data/raw/` and writes `data/db/energyiq.duckdb` (about a minute). Rerun it whenever the raw data changes. The database isn't committed, so everyone builds their own.

This plain build only loads readings and weather; the `anomalies` table is left empty. To also fill in anomalies, build with `--replay` instead (see [Building the database with replays](#building-the-database-with-replays)).

**Always run scripts with `python -m` and dotted names** (e.g. `python -m scripts.build_db`, not `python scripts/build_db.py`). Otherwise you get `No module named 'energyiq'`.

**Close the database in any other tool** (e.g. the VS Code DuckDB extension) before rebuilding, or the build fails with "file is being used by another process."

### Browsing the database (optional)

Install the **DuckDB** VS Code extension, then use **DuckDB: Select Database → Attach** on `data/db/energyiq.duckdb`. Don't double-click the file; that fails on Windows. Detach it before rebuilding.

## Building the database with replays

```bash
python -m scripts.build_db --replay
```

This builds the same database as the plain build, but feeds the data in one time segment at a time, the way the system would have seen it live. Before each new segment is added, the models trained on everything *before* it look for anomalies in it, so no anomaly is ever decided using data from its own future. It takes noticeably longer than the plain build, since it trains two models per segment.

It's the command to use whenever you need the `anomalies` table (dashboard, alerts, anomaly analysis). Each run drops and rebuilds the whole database, so it's safe to rerun.

### What it does, step by step

The segments are listed in `TIME_SEGMENTS` in `energyiq/config.py`, one per folder in `data/raw/`. The replay starts at `REPLAY_INIT_TRAIN_END` (currently `2025-09-01`).

1. **Initial state.** Wipes and recreates every table, then loads all readings and weather *before* `2025-09-01` (Sep 2023 – Aug 2025). No anomalies are detected for this period; it's the starting training data.
2. **Initial models.** Trains one HGB model per utility (`heat`, `electricity`) on all three buildings at once, using the default features from `DEFAULT_FEATURES` and settings from `HGB_DEFAULTS` in `config.py`. Saved to `energyiq/ml/models/replay/2025-09-01/<utility>.joblib`.
3. **For each later segment** (`sept2025-feb2026`, then `march2026-sept2026`):
   1. **Insert readings.** Appends the segment's readings and weather to the database.
   2. **Detect.** `AnomalyDetector` loads the previous step's models and runs them over the segment, reading it from the database along with the 8 days before it so the first week's lag features have real history. The models were trained before the segment was added, and every feature only looks backwards in time, so there's no lookahead. Where a lag is still missing (a gap in the data), it's left empty rather than guessed.
   3. **Insert anomalies.** Appends the segment's anomalies to `anomalies`, tagged with the segment's folder name in the `segment` column.
   4. **Retrain.** Trains fresh models on everything in the database so far, **except readings already flagged as anomalies**, and saves them under `replay/<segment end date>/` (e.g. `replay/2026-03-01/`). These are used to detect on the next segment.

The models from the final retrain (`replay/2026-09-01/`) are trained on all the data and are the "current" models.

Example of what you'll see:

```
============================================================
INITIAL STATE: data before 2025-09-01
============================================================
-- training heat (data before 2025-09-01) --
...
============================================================
BATCH sept2025-feb2026: 2025-09-01 to 2026-03-01
============================================================
-- detecting anomalies with 2025-09-01 models --
-- writing readings, weather and anomalies to the DB --
sept2025-feb2026: <N> anomalies
...
DONE: wrote .../data/db/energyiq.duckdb
```

### How an anomaly is decided

When a replay model is trained, it records its average error (MAE) on its own training data, separately for each **group** of readings:

- **Heat:** per meter and season.
- **Electricity:** per meter, weekday/weekend, and academic break vs. in-session. Ohio Stadium event days get their own group, so game days are compared against other game days rather than normal days.

A new reading is an anomaly when `|actual − predicted|` is more than `ANOMALY_K` (currently **8**) times its group's training MAE. A group never seen in training falls back to the model's overall training MAE. This is what gives us thresholds that change with weekday/weekend, break, and season instead of one fixed cutoff.

### The `anomalies` table

| Column | Meaning |
|---|---|
| `meterid`, `readingtime`, `building_id`, `utility` | Which reading was flagged |
| `actual`, `predicted` | Metered value and the model's prediction |
| `residual` | `actual − predicted` (positive = used more than expected) |
| `threshold` | The cutoff the residual was compared to (`ANOMALY_K` × group MAE) |
| `severity` | `|residual| / threshold`, always > 1. Higher is more unusual |
| `segment` | The replay batch it was found in (data folder name) |
| `alert_sent` | Whether an alert has gone out for it (always false for now) |

Quick check after a build, in the DuckDB extension or Python:

```sql
SELECT b.sitename, a.utility, a.segment, COUNT(*) AS n, ROUND(AVG(a.severity), 2) AS avg_severity
FROM anomalies a JOIN buildings b ON b.id = a.building_id
GROUP BY ALL ORDER BY ALL;
```

### Things to know

- **It overwrites the committed replay models.** The files in `energyiq/ml/models/replay/` are checked into git, and every `--replay` run rewrites them. If you didn't mean to change them (e.g. you only rebuilt to get the database), discard them with `git checkout -- energyiq/ml/models/replay`.
- **Changing the replay** (different `ANOMALY_K`, features, HGB settings, or start date) is done in `energyiq/config.py`, then rerun `--replay`. Mention it in your PR, since everyone's anomalies will change.
- **Adding a new data folder:** add it to `data/raw/` and append a `(folder, start, end)` entry to `TIME_SEGMENTS`. The replay picks it up as another batch automatically.
- Same as the plain build: run it from `backend/` with `python -m`, and close the database in any other tool first.

## Training a model

`train.py` is for experiments: it trains one model, scores it on a held-out test period, and prints the results. (The models the replay uses are trained separately by `build_db --replay`.)

```bash
python -m energyiq.ml.train --building "Ohio Stadium" --utility-type electricity --model-type hgb -v
```

- `--building`: `All` (default, one model across all three buildings), `Ohio Stadium`, `Thompson Memorial Library`, or `Scott House`.
- `--utility-type`: `electricity` (default) or `heat`. Each utility is modeled separately.
- `--model-type`: `hgb` (gradient-boosted trees, the main model) or `regression` (linear baseline, the default).
- `-v`: prints the features used, overall test metrics, and test error broken down per month and per meter.
- `--anomaly-k`: threshold multiplier for flagging test anomalies (default `ANOMALY_K` from `config.py`).
- `--save`: saves the model to `energyiq/ml/models/experiments/<utility>/` and updates `best.joblib` / `best.json` there if its MAE beats the saved best. These experiment models aren't committed.

**Feature flags.** The defaults per utility come from `DEFAULT_FEATURES` in `config.py`, so you don't need to pass anything for the standard setup. Pass a flag to turn a feature on, or its `--no-` form (e.g. `--no-lags`) to turn a default off.

| Flag | Utility | Adds | On by default |
|---|---|---|---|
| `--lags` | both | Usage 24 hours and 1 week earlier | yes |
| `--weekdays` | electricity | Day of the week | yes |
| `--breaks` | electricity | Academic break indicator | yes |
| `--events` | electricity | Ohio Stadium event-day and during-event flags (games, concerts; dates in `STADIUM_EVENTS` in `preprocess.py`) | yes |
| `--weekends` | electricity | Weekend indicator | no |
| `--temp-history` | heat | Recent average/minimum temperatures | yes |
| `--usage-history` | heat | Average usage over a past 24-hour window | yes |
| `--seasons` | heat | Season of the year | yes |
| `--months` | heat | Month of the year | no |

Turning on a heat flag for electricity (or the reverse) gives an error.

## How models are evaluated

- **Train:** everything before `EXPERIMENT_TRAIN_END` (`2025-09-01`), i.e. Sep 2023 – Aug 2025. **Test:** Sep 2025 – Aug 2026. Split by date, never shuffled.
- Training data is cleaned first: runs of zero readings are fixed and extreme outliers are dropped. Test data is left raw.
- Synthetic anomalies (short spikes and dropouts) are injected into the test period with a fixed seed, so the anomaly detector can be scored against known answers. Accuracy metrics are computed on the real (non-injected) test readings.
- The main metric is **MAE** (average error, in the meter's units) on the test period, after trimming the largest 2% of errors (`TRIM_PCT` in `evaluation.py`). Only compare MAE within the same building and utility. Detection metrics for the injected anomalies are reported too.
- HGB settings were picked by a grid search and live in `HGB_DEFAULTS` in `config.py`.
- **Build the database without `--replay` before running experiments.** Training data is read from the database, and readings in the `anomalies` table are left out. After a replay build, that removes flagged readings from the test period, so the scores won't be comparable with runs on a plain build.

## Adding a dependency

1. `pip install <package>` with the venv active.
2. Add it to `backend/requirements.txt` and commit.
3. Teammates run `pip install -r requirements.txt` after pulling.
