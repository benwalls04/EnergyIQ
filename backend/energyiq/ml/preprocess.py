import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from energyiq import config 

def preprocess(building_name, utility_type):
    db_path = config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"duckdb:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)

    def time_bucket(hour):
      if hour >= 22 or hour < 6:
          return "night"
      if hour < 9:
          return "early_morning"
      if hour < 12:
          return "late_morning"
      if hour < 15:
          return "early_afternoon"
      if hour < 19:
          return "late_afternoon"
      return "evening"

    breaks = [
      ("10/12/23", "10/13/23"), 
      ("11/22/23", "11/24/23"),
      ("12/15/23", "1/7/24"),
      ("3/11/24", "3/15/24"),
      ("5/1/24", "8/19/24"),
      ("10/10/24", "10/11/24"), 
      ("11/27/24", "11/29/24"),
      ("12/13/24", "1/5/25"), 
      ("3/10/25", "3/14/25"),
      ("4/30/25", "8/18/25"),
      ("10/16/25", "10/17/25"), 
      ("11/26/25", "11/28/25"),
      ("12/19/25", "1/11/26"),
      ("3/16/26", "3/20/26"),
      ("5/5/26", "8/24/26"),
      ("10/15/26", "10/16/26"),
      ("11/25/26", "11/27/26"),
      ("11/18/26", "1/10/27"), 
      ("3/15/27", "3/19/27"),
      ("5/5/27", "8/23/27")
    ]

    def drop_outliers(df, col="readingvalue", group="meterid", k=10):
      df = df[df[col] >= 0]   # negative usage is impossible

      med = df.groupby(group)[col].transform("median")
      mad = df.groupby(group)[col].transform(lambda s: (s - s.median()).abs().median())
      mad = mad.replace(0, df[col].abs().median())   # avoid dividing by zero for flat meters

      keep = (df[col] - med).abs() <= k * mad
      print(f"dropping {(~keep).sum()} of {len(df)} rows")
      return df[keep]
    
    with SessionLocal.begin() as session:
      result = session.execute(
         text("SELECT id, sitename FROM buildings WHERE sitename =:name"),
         {"name": building_name}
      )
      bId = result.first()[0]

      df = pd.read_sql(text("""
          SELECT m.*, b.sitename, w.*
          FROM meter_readings m
          JOIN buildings b ON m.building_id = b.id
          JOIN weather w
            ON date_trunc('hour', m.readingtime) = date_trunc('hour', w.date)
          WHERE m.building_id = :bId
      """), session.connection(), params={"bId": bId})

    # testing one utility at a time
    if utility_type:
       df = df[df["utility"] == utility_type.upper()]

    df = drop_outliers(df)

    # lag featues - 24 hours ago and 7 hours ago
    df = df.sort_values(["meterid", "readingtime"])
    g = df.groupby("meterid")["readingvalue"]
    df["lag_24h"] = g.shift(96)     # 96 × 15 min = 24 hours
    df["lag_7d"]  = g.shift(672)    # 672 × 15 min = 7 days
    df = df.dropna(subset=["lag_24h", "lag_7d"])

    df = pd.get_dummies(df, columns=["utility", "meterid"], dtype=int)
    df["is_weekend"] = (df["readingtime"].dt.dayofweek >= 5).astype(int)

    df["time_of_day"] = df["readingtime"].dt.hour.map(time_bucket)
    df = pd.get_dummies(df, columns=["time_of_day"], prefix="tod", dtype=int)

    dates = df["readingtime"].dt.normalize()     # drop the time, so the end day counts in full

    on_break = pd.Series(False, index=df.index)
    for start, end in breaks:
        s = pd.to_datetime(start, format="%m/%d/%y")
        e = pd.to_datetime(end, format="%m/%d/%y")
        on_break |= dates.between(s, e)          # inclusive on both ends

    df["in_classes"] = (~on_break).astype(int)
    df = df.drop(columns=["readingunits", "sitename", "latitude", "longitude", "building_id", "date"], errors="ignore")
    df = df.sort_values("readingtime").set_index("readingtime")
    X = df.drop(columns=["readingvalue"])
    Y = df["readingvalue"]

    return X, Y

if __name__=="__main__":
  preprocess("Ohio Stadium")
        
