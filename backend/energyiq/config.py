from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "db" / "energyiq.duckdb"

MODELS_DIR = Path(__file__).resolve().parent / "ml" / "models"
EXPERIMENT_MODELS_DIR = MODELS_DIR / "experiments" 
REPLAY_MODELS_DIR = MODELS_DIR / "replay" 
WEATHER_DIR = "weather_data"  

UTILITIES = ["heat", "electricity"]

HGB_DEFAULTS = {
    "random_state": 42,
    "loss": "absolute_error",
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "early_stopping": False, 
}

ANOMALY_K = 10

TRAIN_END = "2025-09-01"

TIME_SEGMENTS = [
    ("sep2023-feb2024", "2023-09-01", "2024-03-01"),
    ("mar2024-aug2024", "2024-03-01", "2024-09-01"),
    ("sept2024-feb2025", "2024-09-01", "2025-03-01"),
    ("march2025-aug2025", "2025-03-01", "2025-09-01"),
    ("sept2025-feb2026", "2025-09-01", "2026-03-01"),
    ("march2026-sept2026", "2026-03-01", "2026-09-01"),
]

# academic breaks, (first day, last day) inclusive
BREAKS = [
    ("2023-10-12", "2023-10-13"), ("2023-11-22", "2023-11-24"), ("2023-12-15", "2024-01-07"),
    ("2024-03-11", "2024-03-15"), ("2024-05-01", "2024-08-19"),
    ("2024-10-10", "2024-10-11"), ("2024-11-27", "2024-11-29"), ("2024-12-13", "2025-01-05"),
    ("2025-03-10", "2025-03-14"), ("2025-04-30", "2025-08-18"),
    ("2025-10-16", "2025-10-17"), ("2025-11-26", "2025-11-28"), ("2025-12-19", "2026-01-11"),
    ("2026-03-16", "2026-03-20"), ("2026-05-05", "2026-08-24"),
    ("2026-10-15", "2026-10-16"), ("2026-11-25", "2026-11-27"),
    ("2026-12-18", "2027-01-10"),  # was 11/18/26, check against the calendar
    ("2027-03-15", "2027-03-19"), ("2027-05-05", "2027-08-23"),
]

# Ohio Stadium events, (start, approx. end) in Eastern local time. readingtime is UTC, so these
# need converting before matching. Published schedules, so calendar data, not occupancy
STADIUM_EVENTS = [
    # 2023 football
    ("2023-09-09 12:00", "2023-09-09 15:30"),  # vs. Youngstown State
    ("2023-09-16 16:00", "2023-09-16 19:30"),  # vs. Western Kentucky
    ("2023-10-07 12:00", "2023-10-07 15:30"),  # vs. Maryland
    ("2023-10-21 12:00", "2023-10-21 15:30"),  # vs. Penn State
    ("2023-11-11 19:30", "2023-11-11 23:00"),  # vs. Michigan State
    ("2023-11-18 16:00", "2023-11-18 19:30"),  # vs. Minnesota
    # 2024 spring and summer
    ("2024-04-13 12:00", "2024-04-13 14:30"),  # spring game
    ("2024-05-05 12:00", "2024-05-05 15:00"),  # spring commencement
    ("2024-06-22 10:00", "2024-06-22 23:30"),  # Buckeye Country Superfest day 1 (tailgate from 10)
    ("2024-06-23 15:00", "2024-06-23 23:00"),  # Buckeye Country Superfest day 2
    # 2024 football
    ("2024-08-31 15:30", "2024-08-31 19:00"),  # vs. Akron
    ("2024-09-07 19:30", "2024-09-07 23:00"),  # vs. Western Michigan
    ("2024-09-21 12:00", "2024-09-21 15:30"),  # vs. Marshall
    ("2024-10-05 15:30", "2024-10-05 19:00"),  # vs. Iowa
    ("2024-10-12 08:00", "2024-10-12 21:00"),  # Buckeye Invitational (high school bands)
    ("2024-10-26 12:00", "2024-10-26 15:30"),  # vs. Nebraska
    ("2024-11-09 12:00", "2024-11-09 15:30"),  # vs. Purdue
    ("2024-11-23 12:00", "2024-11-23 15:30"),  # vs. Indiana
    ("2024-11-30 12:00", "2024-11-30 15:45"),  # vs. Michigan
    ("2024-12-21 20:00", "2024-12-21 23:30"),  # CFP first round vs. Tennessee
    # 2025 spring and summer
    ("2025-03-01 16:00", "2025-03-01 21:00"),  # NHL Stadium Series (gates 4, puck drop 6)
    ("2025-04-12 12:00", "2025-04-12 14:30"),  # spring game
    ("2025-05-04 12:00", "2025-05-04 15:00"),  # spring commencement
    ("2025-06-21 11:00", "2025-06-21 23:00"),  # Buckeye Country Superfest (fan fest from 11)
    # 2025 football
    ("2025-08-30 12:00", "2025-08-30 15:30"),  # vs. Texas
    ("2025-09-06 15:30", "2025-09-06 19:00"),  # vs. Grambling State
    ("2025-09-13 19:00", "2025-09-13 22:30"),  # vs. Ohio
    ("2025-10-04 15:30", "2025-10-04 19:00"),  # vs. Minnesota
    ("2025-10-11 08:00", "2025-10-11 21:00"),  # Buckeye Invitational (high school bands)
    ("2025-11-01 12:00", "2025-11-01 15:30"),  # vs. Penn State
    ("2025-11-15 15:30", "2025-11-15 19:00"),  # vs. UCLA
    ("2025-11-22 12:00", "2025-11-22 15:30"),  # vs. Rutgers
    # 2026 spring and summer
    ("2026-04-18 12:00", "2026-04-18 14:30"),  # spring game
    ("2026-04-25 17:45", "2026-04-25 21:45"),  # Luke Combs
    ("2026-05-03 12:00", "2026-05-03 15:00"),  # spring commencement
    ("2026-05-20 19:00", "2026-05-20 23:00"),  # Bruno Mars
    ("2026-06-13 14:30", "2026-06-13 23:00"),  # Buckeye Country Superfest
    ("2026-07-15 19:30", "2026-07-15 23:30"),  # AC/DC
    # 2026 football
    ("2026-09-05 12:30", "2026-09-05 16:00"),  # vs. Ball State
    ("2026-09-19 12:00", "2026-09-19 15:30"),  # vs. Kent State
    ("2026-09-26 12:00", "2026-09-26 15:30"),  # vs. Illinois
    ("2026-10-03 08:00", "2026-10-03 21:00"),  # Buckeye Invitational (high school bands)
    ("2026-10-10 16:15", "2026-10-10 19:45"),  # vs. Maryland
    ("2026-10-17 16:00", "2026-10-17 23:00"),  # Twenty One Pilots (gates 4, show 6)
]
