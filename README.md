# Campus EnergyIQ

Predicts each campus building's expected hourly energy use from weather and calendar data, then flags hours where actual usage is unusually far from that prediction. Buildings: Ohio Stadium, Thompson Memorial Library, Scott House.

## Repo layout

```
EnergyIQ/
├── backend/
│   ├── energyiq/
│   │   ├── config.py        # file paths (raw data folder, database location)
│   │   ├── models.py        # database table definitions
│   │   ├── ingest.py        # loads the raw CSVs into the database
│   │   └── ml/
│   │       ├── preprocess.py  # turns database rows into model features
│   │       ├── train.py       # trains and scores one model
│   │       ├── sweep.py       # runs many training configs, logs them to W&B
│   │       └── models/        # saved models, one folder per building/utility
│   ├── scripts/
│   │   └── build_db.py      # builds the database from the raw CSVs
│   └── requirements.txt
└── data/
    ├── raw/                 # source CSVs, one folder per date range + weather_data/
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

**Always run scripts with `python -m` and dotted names** (e.g. `python -m scripts.build_db`, not `python scripts/build_db.py`). Otherwise you get `No module named 'energyiq'`.

**Close the database in any other tool** (e.g. the VS Code DuckDB extension) before rebuilding, or the build fails with "file is being used by another process."

### Browsing the database (optional)

Install the **DuckDB** VS Code extension, then use **DuckDB: Select Database → Attach** on `data/db/energyiq.duckdb`. Don't double-click the file; that fails on Windows. Detach it before rebuilding.

## Training a model

```bash
python -m energyiq.ml.train "Ohio Stadium" --utility-type electricity --model-type hgb --lags --weekends -v
```

- `--utility-type`: `electricity` or `heat`. Each utility is modeled separately.
- `--model-type`: `hgb` (gradient-boosted trees, the main model) or `regression` (linear baseline).
- `-v`: prints the features used, overall test metrics, and test error per month and per meter.
- `--save`: saves the model to `energyiq/ml/models/<building>/<utility>/` and updates `best.joblib` if it beats the saved best.

Feature flags (all off unless passed):

| Flag | Utility | Adds |
|---|---|---|
| `--lags` | both | Usage 24 hours and 1 week earlier |
| `--weekends` | electricity | Weekend indicator |
| `--breaks` | electricity | Academic break indicator |
| `--temp-history` | heat | Recent average/minimum temperatures |
| `--usage-history` | heat | Average usage over a past 24-hour window |
| `--seasons` | heat | Season of the year |

Using a heat flag with electricity (or the reverse) gives an error.

Current picks: electricity uses `--lags --weekends` for Ohio Stadium, and `--lags --weekends --breaks` for the other two buildings.

## How models are evaluated

- **Train:** Sep 2024 – Aug 2025. **Test:** Sep 2025 – Aug 2026. One full year each, split by date (never shuffled).
- The test year is also the period the anomaly replay will run over.
- The main metric is **MAE** (average error in kWh) on the test year. Only compare MAE within the same building and utility.
- HGB settings were picked by a grid search and are the defaults in `train.py` (`HGB_DEFAULTS`).

## Running sweeps (W&B)

`sweep.py` trains many configurations and logs each one as a run in Weights & Biases.

One-time setup: `wandb login` (key from https://wandb.ai/authorize).

```bash
python -m energyiq.ml.sweep "Ohio Stadium" --stage 0 --dry-run            # list planned runs, train nothing
python -m energyiq.ml.sweep "Ohio Stadium" --stage 0 --limit 3 --project energyiq-test   # quick test
python -m energyiq.ml.sweep "Ohio Stadium" --stage 0                      # full run
```

- **Stage 0** compares feature combinations. **Stage A** compares HGB settings.
- Runs land in the `energyiq` project, grouped as `stage<N>/<building>` and tagged with the building and stage.
- There's no resume: if a sweep stops partway, rerun it.

## Adding a dependency

1. `pip install <package>` with the venv active.
2. Add it to `backend/requirements.txt` and commit.
3. Teammates run `pip install -r requirements.txt` after pulling.
