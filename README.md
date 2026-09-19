# Campus EnergyIQ

## Environment setup

### Prerequisites

- **Python 3.10 or higher** installed on your machine.
  - Check your version with:

    ```bash
    python --version
    ```

  - If that command isn't found, try `python3 --version` (common on macOS/Linux) or `py --version` (Windows).
  - If you don't have 3.10+, install it from [python.org](https://www.python.org/downloads/) or your package manager.

The exact patch version (3.10.x vs 3.12.x) doesn't need to match across the team — just stay on 3.10+ so everyone's dependencies install and behave consistently.

### 1. Clone the repo and move into it

```bash
git clone <repo-url>
cd EnergyIQ
```

### 2. Create a virtual environment

This creates an isolated Python environment local to your machine, in a `.venv` folder. **`.venv` is gitignored — do not commit it.** Each teammate creates their own.

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

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Deactivate when you're done

```bash
deactivate
```

## Populating your local database

The DuckDB database (`db/energyiq.duckdb`) is **not committed to git** — it's a generated file (multiple GB once built) that's gitignored (`*.duckdb`, `*.duckdb.wal`). Each teammate builds their own local copy from the raw CSVs in `data/`, which *are* committed.

With your virtual environment activated (steps 2-4 above), run this from anywhere inside the repo:

```bash
python scripts/ingestion.py
```

This reads all the CSVs under `data/` and `data/weather_data/`, and writes `buildings`, `meter_readings`, and `weather` tables into `db/energyiq.duckdb`. It takes a minute or two. Re-run it any time you pull new/changed data — it drops and rebuilds all three tables from scratch.

### Browsing the database in VS Code

If you use the **DuckDB** extension (`chuckjonas.duckdb`) to inspect the file, don't open `db/energyiq.duckdb` directly (double-click / "Open With" → its table viewer) — on Windows this mis-parses the file path and fails with `Catalog "c:" does not exist!`. Instead:

1. Command Palette → **"DuckDB: Select Database"** (or the duck icon in the Activity Bar)
2. Choose **Attach** and point it at `db/energyiq.duckdb`
3. Browse tables from the sidebar tree once it's attached

## Adding a new dependency

If your work needs a new package:

1. With your venv activated, install it: `pip install <package-name>`
2. Add it to `requirements.txt` with a version constraint (check what got installed via `pip show <package-name>`), or regenerate the whole file with `pip freeze > requirements.txt` if you want everything pinned exactly.
3. Commit the updated `requirements.txt` so the rest of the team can pick it up.
4. Teammates pull the change and run `pip install -r requirements.txt` again to sync.
