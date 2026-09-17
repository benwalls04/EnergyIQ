# script that dumps our data to database

import os
import pathlib

import pandas as pd


def find_project_root(marker: str = ".git") -> pathlib.Path:
    """Walk upward from the current working directory until a folder containing `marker` is found."""
    current = pathlib.Path.cwd()
    for candidate in [current, *current.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find project root (no '{marker}' found in {current} or any parent directory)"
    )


PROJECT_ROOT = find_project_root()
print(f"Project root: {PROJECT_ROOT}")

data_path = PROJECT_ROOT / "data"
time_segments = ["sept2024-feb2025", "march2025-aug2025", "sept2025-feb2026", "march2026-sept2026"]

all_dfs = []
for subfolder_name in time_segments:
    subfolder_path = data_path / subfolder_name
    csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = subfolder_path / csv_file
        df = pd.read_csv(file_path)

        print(f"Data from {file_path}:")
        print(df.head())

        all_dfs.append(df)

# Combine everything into one DataFrame, if that's the goal
combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()