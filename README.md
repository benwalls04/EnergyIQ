# Campus EnergyIQ

## Repo layout

```
EnergyIQ/
├── backend/
│   ├── energyiq/          # Python package: config, models, ingest
│   ├── scripts/
│   │   └── build_db.py    # builds the DuckDB database from the raw CSVs
│   ├── tests/
│   └── requirements.txt
└── data/
    ├── raw/               # source CSVs, one folder per date range + weather_data/ (committed)
    └── db/                # energyiq.duckdb is written here (gitignored)
```

## Backend setup

### Prerequisites

- **Python 3.10 or higher** installed on your machine.
  - Check your version with:

    ```bash
    python --version
    ```

  - If that command isn't found, try `python3 --version` (common on macOS/Linux) or `py --version` (Windows).
  - If you don't have 3.10+, install it from [python.org](https://www.python.org/downloads/) or your package manager.

The exact patch version (3.10.x vs 3.12.x) doesn't need to match across the team — just stay on 3.10+ so everyone's dependencies install and behave consistently.

### 1. Clone the repo and move into the backend folder

```bash
git clone <repo-url>
cd EnergyIQ/backend
```

**All backend commands below are run from inside `backend/`.**

### 2. Create a virtual environment

This creates an isolated Python environment local to your machine, in `backend/.venv`. **`.venv` is gitignored — do not commit it.** Each teammate creates their own.

```bash
python -m venv .venv
```

(If `python` isn't recognized, substitute `python3` or `py` as noted above.)

### 3. Activate the virtual environment

You need to do this every time you open a new terminal to work on the project.

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

If you get a "running scripts is disabled" error, run this once (in an admin PowerShell) to allow local scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Your terminal prompt should now be prefixed with `(.venv)`.

In VS Code, also point the editor at this environment: Command Palette → **"Python: Select Interpreter"** → pick `backend/.venv`. Otherwise imports show as unresolved and the Run button uses the wrong Python.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Build the database

The DuckDB database (`data/db/energyiq.duckdb`) is **not committed to git** — it's a generated file that's gitignored (`*.duckdb`, `*.duckdb.wal`). Each teammate builds their own local copy from the raw CSVs in `data/raw/`, which *are* committed.

With your virtual environment activated, from inside `backend/`:

```bash
python -m scripts.build_db
```

Note the `-m` and the dotted name with **no `.py`** — running it as a module is what lets the script import the `energyiq` package. Running `python scripts/build_db.py` directly fails with `ModuleNotFoundError: No module named 'energyiq'`, and so does running it from any folder other than `backend/`.

This reads every CSV under `data/raw/<date-range>/` and `data/raw/weather_data/`, and writes `buildings`, `meter_readings`, and `weather` tables into `data/db/energyiq.duckdb`. It takes a minute or two. Re-run it any time you pull new/changed data — it drops and rebuilds all three tables from scratch.

**Close/detach the database in any other tool (e.g. the VS Code DuckDB extension) before rebuilding.** DuckDB only allows one process to hold the file open for writing, so the build fails with `The process cannot access the file because it is being used by another process` if it's attached elsewhere.

### 6. Install the DuckDB extension for VS Code

To browse the database, install the **DuckDB** extension (`chuckjonas.duckdb`) from the VS Code Extensions view.

Don't open `data/db/energyiq.duckdb` directly (double-click / "Open With" → its table viewer) — on Windows this mis-parses the file path and fails with `Catalog "c:" does not exist!`. Instead:

1. Command Palette → **"DuckDB: Select Database"** (or the duck icon in the Activity Bar)
2. Choose **Attach** and point it at `data/db/energyiq.duckdb`
3. Browse tables from the sidebar tree once it's attached

### 7. Verify the database is populated

With the database attached in the extension, run these queries:

```sql
SELECT * FROM buildings;
```

You should get exactly 3 rows: Ohio Stadium (`082`), Thompson Memorial Library (`050`), and Scott House (`1108`). The SIMS codes should keep their leading zeros — if you see `82` or `50`, your copy of `energyiq/ingest.py` is out of date.

```sql
SELECT COUNT(*), MIN(readingtime), MAX(readingtime) FROM meter_readings;
SELECT COUNT(*), MIN(date), MAX(date) FROM weather;
```

Both counts should be non-zero, and both date ranges should span roughly **September 2024 → September 2026**. A range that stops early usually means one of the `data/raw/` folders is missing or empty.

When you're done, **detach** the database in the extension before running `build_db` again (see step 5).

### Deactivate when you're done

```bash
deactivate
```

## Adding a new dependency

If your work needs a new package:

1. With your venv activated, install it: `pip install <package-name>`
2. Add it to `backend/requirements.txt` with a version constraint (check what got installed via `pip show <package-name>`), or regenerate the whole file with `pip freeze > requirements.txt` (from inside `backend/`) if you want everything pinned exactly.
3. Commit the updated `backend/requirements.txt` so the rest of the team can pick it up.
4. Teammates pull the change and run `pip install -r requirements.txt` again (from inside `backend/`) to sync.
