import argparse
import subprocess
import json
import pathlib
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from energyiq import config
from energyiq.models import Base, Event

PROMPT = """
The candidates cover one building and one utility over one calendar month. A header at the end of this prompt names
the building (sitename and simscode), the utility, the units of readingvalue and the dates covered. The CSV
after it has one row per flagged reading:
- meterid, readingtime: together identify a reading
- readingvalue: what was measured, in the units from the header
- severity: how far the reading fell outside its expected range, in multiples of that range. Above 1
  means usage was higher than expected, below -1 means lower than expected.
- temperature_2m: outdoor temperature (Celsius) at that hour

Grouping rules:
1) An event covers one building and one utility. Group readings from several meters in the building only when they deviate at the same time.
2) Readings are part of the same event when consecutive readings are at most 6 hours apart and deviate in the same direction (all above or all below expected).
3) Each reading goes in at most one event.
4) The only exception to rule 2: when the same meter deviates in the same direction at roughly the same time of day
   (start and end times within about two hours of each other) on most days over a span of days, merge those daily
   runs into one recurring event, even though they aren't continuous. Skipped days, such as weekends, don't break
   the pattern. A recurring event's start_time and end_time are its first and last readings, its readings are every
   reading in each daily run, and its description says that it recurs, at what time of day, and on which dates.

Severity:
Judge each event's severity yourself; there are no fixed cutoffs. Meters behave very differently, so
compare the event with this meter's own flagged readings in the CSV, not with an absolute scale. Weigh:
- how far the readings went beyond the expected range, and whether that is large for this meter
- how long the event lasted, and how steadily the readings stayed outside the range
- whether an ordinary explanation fits (a game day, a cold snap, a holiday break); a likely ordinary
  explanation lowers severity
Tiers:
- low: unusual, but probably harmless; worth knowing about
- medium: unusual enough that someone should check on it
- high: clearly abnormal for this meter; needs attention soon
Decide severity from the event itself before applying the inclusion rule below. Don't raise or lower it
to change whether the event is included.

Inclusion rule (applied after severity is decided):
An event's span is end_time minus start_time; a one-reading event spans 0 hours.
- low: include only if it spans at least 2 days
- medium: include only if it spans at least 6 hours
- high: always include
Leave out events that don't qualify.

Neutrality:
There is no target number of events, and no goal to filter readings out or to group them in. Judge each
reading and event on the data alone. Returning no events, or grouping every reading into events, are
both correct when the data supports them.

Possible causes (others are fine):
- Ohio Stadium electricity spikes during home football games, concerts or other events, usually on weekends. Ohio Stadium home games occur on Saturdays in the fall. Concerts typically happen on weekends any time of the year.  
- Heating spikes during winter cold fronts (check temperature_2m)
- A very large spike on one meter, suggesting an equipment fault
- A meter slightly above or below its range for days or longer, suggesting gradual equipment drift
Still report events that have an obvious explanation, such as a game day, and name that explanation.
You can't check event calendars or maintenance logs, so present every cause as possible, not confirmed.

Each event is sent as its own email to a facilities manager, who sees only that one email: not the CSV,
not the other events, and not how events were formed. In the title, description and likely_causes:
- Don't mention candidates, the CSV, grouping or gap rules, other events, or how this event compares to them.
- Don't label the event's severity (e.g. "low severity").
- Use the building's name, not its simscode, and say "outdoor temperature", not temperature_2m.
- Write plain dates and times, and durations in hours or days.

Example description: "Cooling use at Thompson Library (meter 247288) ran above its expected range from midnight to 4:30 AM on November 16, 2025, peaking at about 1.6 times the expected range at midnight. The outdoor temperature was unusually warm for a November night, which may explain part of the extra load."

Fields for each event:
- title: one line
- description: what was observed and its scope (building, meters, how far above or below expected, for how long)
- likely_causes: the possible causes above that fit, or others you can justify from the data
- severity: low, medium or high, from your judgment above
- severity_reason: one sentence explaining the severity, written for the facilities manager. It may refer
  to the meter's usual behavior, but not to other events or the CSV.
- utility, building: as in the header, with building as the simscode
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
        "severity_reason": {"type": "string"},
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
    "required": ["title", "description", "likely_causes", "severity", "severity_reason",
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
        r"C:\Users\benwa\AppData\Roaming\npm\node_modules\@anthropic-ai\claude-code\bin\claude.exe",
        "-p",
        "--model", "claude-haiku-5-5",
        "--system-prompt", "You are the last step of a campus building energy anomaly detection pipeline. A model predicts each meter's expected usage from weather and calendar features, and every reading outside its expected range was flagged as an anomaly candidate. Group these candidates into events for a facilities team.",
        "--tools", "",
        "--strict-mcp-config",
        "--output-format", "json",
        "--json-schema", custom_schema
    ]

    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(f"Claude error: {result.stderr}")
    return json.loads(result.stdout)["structured_output"]["events"]


CSV_COLS = ["meterid", "readingtime", "readingvalue", "severity", "temperature_2m"]


def build_prompt(group, window_start):
    """The prompt for one batch: the columns that are the same on every row go in a header, the rest in the CSV."""
    first = group.iloc[0]
    window_end = window_start + pd.offsets.MonthEnd(0)
    header = (f"Building: {first.sitename} (simscode {first.simscode})\n"
              f"Utility: {first.utility}\n"
              f"Units: {first.readingunits}\n"
              f"Dates: {window_start:%Y-%m-%d} to {window_end:%Y-%m-%d}\n\n")
    return PROMPT + header + group[CSV_COLS].to_csv(index=False)


MIN_SPAN = {"low": timedelta(days=2), "medium": timedelta(hours=6)}


def meets_duration(event):
    span = datetime.fromisoformat(event["end_time"]) - datetime.fromisoformat(event["start_time"])
    return event["severity"] == "high" or span >= MIN_SPAN[event["severity"]]


EVENT_COLS = ["title", "description", "likely_causes", "severity", "severity_reason",
              "utility", "building", "start_time", "end_time"]


def save_events(log):
    """Replaces the events table with the events in the log, links their readings, and unflags the rest."""
    events = [e for batch in log.values() for e in batch]
    events_df = pd.DataFrame(events, columns=EVENT_COLS)
    for col in ["start_time", "end_time"]:
        events_df[col] = pd.to_datetime(events_df[col], format="ISO8601", utc=True).dt.tz_localize(None)

    engine = create_engine(f"duckdb:///{config.DB_PATH}")
    try:
        Base.metadata.create_all(engine, tables=[Event.__table__])
        with Session(engine) as session:
            duck = session.connection().connection.driver_connection
            duck.execute("ALTER TABLE meter_readings ADD COLUMN IF NOT EXISTS event_id INTEGER")

            ids = duck.execute("SELECT nextval('seq_event_id') FROM range(?)", [len(events_df)]).fetchall()
            events_df["id"] = [row[0] for row in ids]
            links_df = pd.DataFrame(
                [(r["meterid"], r["readingtime"], event_id)
                 for e, event_id in zip(events, events_df["id"]) for r in e["readings"]],
                columns=["meterid", "readingtime", "event_id"],
            )
            links_df["readingtime"] = pd.to_datetime(links_df["readingtime"], format="ISO8601", utc=True).dt.tz_localize(None)
            claimed = len(links_df)
            links_df = links_df.drop_duplicates(["meterid", "readingtime"], keep="last")

            duck.register("events_df", events_df)
            duck.register("links_df", links_df)
            try:
                duck.execute("UPDATE meter_readings SET event_id = NULL WHERE event_id IS NOT NULL")
                duck.execute("DELETE FROM events")
                inserted = duck.execute("""
                    INSERT INTO events (id, title, description, likely_causes, severity, severity_reason,
                                        utility, building_id, start_time, end_time)
                    SELECT d.id, d.title, d.description, d.likely_causes, d.severity, d.severity_reason,
                           d.utility, b.id, d.start_time, d.end_time
                    FROM events_df d
                    JOIN buildings b ON b.simscode = d.building
                """).fetchone()[0]
                linked = duck.execute("""
                    UPDATE meter_readings AS m
                    SET event_id = l.event_id
                    FROM links_df l
                    WHERE m.meterid = l.meterid AND m.readingtime = l.readingtime
                      AND m.is_anomalous AND l.event_id IN (SELECT id FROM events)
                """).fetchone()[0]
                unflagged = duck.execute("""
                    UPDATE meter_readings
                    SET is_anomalous = FALSE, severity = NULL
                    WHERE is_anomalous AND event_id IS NULL
                """).fetchone()[0]
            finally:
                duck.unregister("events_df")
                duck.unregister("links_df")

            session.commit()
    finally:
        engine.dispose()

    print(f"inserted {inserted} of {len(events_df)} events (the rest had an unknown building)")
    print(f"linked {linked} readings, skipped {claimed - linked} (missing, not flagged, or claimed twice)")
    print(f"unflagged {unflagged} readings with no event")


if __name__ == "__main__":
    candidates = load_candidates()
    print(f"{len(candidates)} anomaly candidates")

    candidates["window"] = candidates["readingtime"].dt.to_period("M").dt.start_time
    parser = argparse.ArgumentParser(description="Group anomaly candidates into events with Claude")
    parser.add_argument("--resume", type=pathlib.Path,
                        help="log file from an earlier run to continue; batches already in it are skipped")
    args = parser.parse_args()

    # the log maps each batch to its kept events, so a batch with no events still counts as done
    if args.resume:
        out_path = args.resume
        done = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"resuming {out_path}: {len(done)} batches already done")
    else:
        log_dir = pathlib.Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        out_path = log_dir / f"emails_{datetime.now():%Y%m%d_%H%M%S}.json"
        done = {}

    for (simscode, utility, window_start), group in candidates.groupby(["simscode", "utility", "window"]):
        key = f"{simscode} {utility} {window_start:%Y-%m-%d}"
        if key in done:
            continue
        print(f"{key}: {len(group)} candidates")
        batch = run_headless(build_prompt(group, window_start))
        kept = [e for e in batch if meets_duration(e)]
        print(f"  kept {len(kept)} events, dropped {len(batch) - len(kept)} that broke the inclusion rule")
        done[key] = kept
        out_path.write_text(json.dumps(done, indent=2), encoding="utf-8")

    print(f"wrote {sum(len(events) for events in done.values())} events from {len(done)} batches to {out_path}")

    save_events(done)