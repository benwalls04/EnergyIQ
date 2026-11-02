import subprocess
import json
import pathlib
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text
from energyiq import config

PROMPT = """
You are the last step of a campus building energy anomaly detection pipeline. A model predicts each
meter's expected usage from weather and calendar features, and every reading outside its expected
range was flagged as an anomaly candidate. Group these candidates into events for a facilities team.

The candidates are in the CSV at the end of this prompt, one row per flagged reading:
- meterid, readingtime: together identify a reading
- sitename, simscode: the building
- utility, readingvalue, readingunits: what was measured
- severity: how far the reading fell outside its expected range, in multiples of that range. Above 1
  means usage was higher than expected, below -1 means lower than expected.
- temperature_2m: outdoor temperature (Celsius) at that hour

Grouping rules:
1) An event covers one building and one utility. Group readings from several meters in the building
   only when they deviate at the same time.
2) Readings are part of the same event when consecutive readings are at most 6 hours apart and deviate
   in the same direction (all above or all below expected).
3) A single reading can be its own event only if |severity| >= 3. A weaker reading that fits no larger
   event is a false positive: leave it out.
4) Each reading goes in at most one event.

Possible causes (others are fine):
- Ohio Stadium electricity spikes during home football games, concerts or other events
- Heating spikes during winter cold fronts (check temperature_2m)
- A very large spike on one meter, suggesting an equipment fault
- A meter slightly above or below its range for days or longer, suggesting gradual equipment drift
Still report events that have an obvious explanation, such as a game day, and name that explanation.
You can't check event calendars or maintenance logs, so present every cause as possible, not confirmed.

Occupancy data isn't available. When a change in how many people were in the building could explain
an event, set occupancy_may_explain to true and say in the description that occupancy couldn't be checked.

Fields for each event:
- title: one line
- description: what was observed and its scope (building, meters, how far above or below expected, for how long)
- likely_causes: the possible causes above that fit, or others you can justify from the data
- severity: from the event's largest |severity|: "low" under 3, "medium" 3 to 6, "high" over 6
- utility, building: as in the CSV, with building as the simscode
- meterid: only when the event covers a single meter
- start_time, end_time: the first and last readingtime in the event
- readings: every reading in the event, as meterid and readingtime

Candidates:
"""

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "likely_causes": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "occupancy_may_explain": {"type": "boolean"},
        "utility": {"type": "string"},
        "building": {"type": "string"},
        "meterid": {"type": "integer"},
        "start_time": {"type": "string", "format": "date-time"},
        "end_time": {"type": "string", "format": "date-time"},
        "readings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "meterid": {"type": "integer"},
                    "readingtime": {"type": "string", "format": "date-time"}
                },
                "required": ["meterid", "readingtime"]
            }
        }
    },
    # meterid is left out when an event spans several meters
    "required": ["title", "description", "likely_causes", "severity", "occupancy_may_explain",
                 "utility", "building", "start_time", "end_time", "readings"]
}

custom_schema = json.dumps({
    "type": "object",
    "properties": {"events": {"type": "array", "items": EVENT_SCHEMA}},
    "required": ["events"]
})


def load_candidates():
    """Every flagged reading, with its building and the outdoor temperature at that hour."""
    engine = create_engine(f"duckdb:///{config.DB_PATH}")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT m.meterid, m.readingtime, b.sitename, b.simscode, m.utility,
                       m.readingvalue, m.readingunits, m.severity, w.temperature_2m
                FROM meter_readings m
                JOIN buildings b ON m.building_id = b.id
                LEFT JOIN weather w ON w.date = date_trunc('hour', m.readingtime)
                WHERE m.is_anomalous
                ORDER BY b.simscode, m.utility, m.meterid, m.readingtime
            """), conn)
    finally:
        engine.dispose()
    # fewer digits = fewer prompt tokens
    return df.round({"readingvalue": 2, "severity": 2, "temperature_2m": 1})


def run_headless(prompt: str):
    # -p runs non-interactively; the prompt goes through stdin since the CSV can pass Windows' command-line length limit
    cmd = [
        "claude",
        "-p",
        "--output-format", "json",
        "--json-schema", custom_schema
    ]

    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(f"Claude error: {result.stderr}")
    return json.loads(result.stdout)["structured_output"]["events"]


if __name__ == "__main__":
    candidates = load_candidates()
    print(f"{len(candidates)} anomaly candidates")
    # events never span buildings or utilities, so each pair gets its own smaller call
    events = []
    for (simscode, utility), group in candidates.groupby(["simscode", "utility"]):
        print(f"{simscode} {utility}: {len(group)} candidates")
        events += run_headless(PROMPT + group.to_csv(index=False))

    log_dir = pathlib.Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / f"emails_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"wrote {len(events)} events to {out_path}")