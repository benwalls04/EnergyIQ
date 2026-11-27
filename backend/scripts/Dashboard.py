import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import os
import duckdb

# DATA PROCESSING FLOW
# --------------------
# CSV Input
# ↓
# Validate required columns
# ↓
# Pull predictions from DB
# ↓
# Clean and convert data types
# ↓
# Filter by utility type
# ↓
# Combine duplicate timestamps
# ↓
# Aggregate by Month / Day / Hour
# ↓
# Display charts and summary tables

st.set_page_config(page_title="Energy Usage Dashboard", layout="wide")

# ----------------------------------------------------
# FORMATTING
# ----------------------------------------------------

def format_short(value):
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}k"
    else:
        return f"{value:.0f}"


def format_axis(value, pos):
    return f"{value:,.0f}"

formatter = FuncFormatter(format_axis)

def display_kpis(df, usage_col):
    total_usage = df[usage_col].sum()
    avg_usage = df[usage_col].mean()
    max_usage = df[usage_col].max()
    reading_count = len(kpi_df)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total Usage (kWh)", f"{total_usage:,.0f}")
    kpi2.metric("Average Hourly Usage (kWh)", f"{avg_usage:,.2f}")
    kpi3.metric("Peak Reading (kWh)", f"{max_usage:,.2f}")
    kpi4.metric("Readings", f"{reading_count:,}")
    
def style_chart(ax, fig):
    fig.patch.set_facecolor("#DCDCDC")
    ax.set_facecolor("#F7F7F7")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ----------------------------------------------------
# COLUMN DEFINITIONS
# ----------------------------------------------------

usage_col = "Total_Usage"
date_col = "readingtime"
pred_col = "Predicted_Usage"

# ----------------------------------------------------
# LOAD AND CACHE DATA
# ----------------------------------------------------

@st.cache_data
def load_data(data_folder):

    csv_files = list(data_folder.rglob("*.csv"))

    if not csv_files:
        return None, 0

    dataframes = []

    required_columns = ["sitename", date_col, "readingvalue", "utility"]

    for csv_file in csv_files:

        temp_df = pd.read_csv(csv_file)

        missing = [c for c in required_columns if c not in temp_df.columns]

        if missing:
            continue

        dataframes.append(temp_df)

    df = pd.concat(dataframes, ignore_index=True)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df["readingvalue"] = pd.to_numeric(df["readingvalue"], errors="coerce")

    df = df.dropna(subset=[date_col, "readingvalue"])

    # Treat negative readings as zero
    df["readingvalue"] = df["readingvalue"].clip(lower=0)

    return df, len(csv_files)


project_folder = Path(__file__).resolve().parents[2]
data_folder = project_folder / "data/raw"

df, file_count = load_data(data_folder)

if df is None:
    st.error("No CSV files found in Data directory")
    st.stop()

st.sidebar.success(f"Loaded {file_count} CSV file(s)")

# ------------------------------------------------
# PULL PREDICTIONS FROM DB
# ------------------------------------------------

con = duckdb.connect(project_folder / "data/db/energyiq.duckdb")

predicted_df = con.execute("""
    SELECT DISTINCT
        meterid,
        readingtime,
        predictedvalue
    FROM meter_readings
    WHERE predictedvalue IS NOT NULL
""").fetch_df()

con.close()

df["meterid"] = df["meterid"].astype(str)
predicted_df["meterid"] = predicted_df["meterid"].astype(str)

df[date_col] = pd.to_datetime(df[date_col])
predicted_df[date_col] = pd.to_datetime(predicted_df[date_col])

df = df.merge(predicted_df, on=["meterid", date_col], how="left")

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------

st.sidebar.header("Filters")
site_name = "Multiple Buildings"

building_options = ["All Buildings"] + sorted(df["sitename"].dropna().unique().tolist())
selected_building = st.sidebar.selectbox("Building", building_options)

meter_lookup = df.groupby("meterid", as_index=False).agg(utility=("utility", "first"))
meter_options = {"All Meters": None}
selected_label = st.sidebar.selectbox("Meter", list(meter_options.keys()))
selected_meter = meter_options[selected_label]

years = sorted(df[date_col].dt.year.dropna().unique().tolist())
selected_year = st.sidebar.selectbox("Year", ["All Years"] + [str(y) for y in years])

utility_options = ["All Utilities"] + sorted(df["utility"].dropna().unique().tolist())
selected_utility = st.sidebar.selectbox("Utility Type", utility_options)

if selected_building != "All Buildings":
    df = df[df["sitename"] == selected_building]

for _, row in meter_lookup.iterrows():
    label = f"{row['utility']} - {row['meterid']}"
    meter_options[label] = str(row["meterid"])

if selected_meter is not None:
    df = df[df["meterid"].astype(str) == selected_meter]

if selected_year != "All Years":
    df = df[df[date_col].dt.year == int(selected_year)]

if selected_utility != "All Utilities":
    df = df[df["utility"] == selected_utility]

# Map user-friendly names shown in the sidebar

aggregation_options = {
    "Average (Hourly)": "mean",
    "Total": "sum",
    "Maximum": "max",
    "Minimum": "min",
}

aggregation_label = st.sidebar.selectbox(
    "Aggregation", list(aggregation_options.keys())
)

agg_function = aggregation_options[aggregation_label]

exclude_outliers = st.sidebar.checkbox(
    "Exclude Extreme Outliers",
    value=True,
    help="Removes unusually large readings that may be caused by bad meter data.",
)

show_predictions = st.sidebar.checkbox(
    "Display Prediction Overlay",
    value=True,
    help="Display predicted usage values alongside actual readings."
)

# Add logo, title, and subheaders

display_column = f"{aggregation_label} Usage (kWh)"

header_col1, header_col2 = st.columns([1, 5])

with header_col1:
    st.image(project_folder / "Logo.png", width=120)

with header_col2:
    st.title("Energy Usage Dashboard")
    st.subheader(selected_building)

# Combine duplicate timestamps and group by hour

group_fields = [date_col]

df["Hour"] = pd.to_datetime(df[date_col]).dt.floor("h")

df = df.groupby(["sitename", "Hour"], as_index=False).agg(
    Total_Usage=("readingvalue", "sum"),
    Predicted_Usage=("predictedvalue", "sum")
)

df = df.rename(columns={"Hour": date_col})

outlier_count = 0

if exclude_outliers:

    upper_bound = df[usage_col].quantile(0.999)

    outlier_count = (df[usage_col] > upper_bound).sum()

    df = df[df[usage_col] <= upper_bound]

if outlier_count > 0:
    st.warning(
        f"⚠ {outlier_count:,} reading(s) above "
        f"{upper_bound:,.2f} kWh were excluded."
    )

if df.empty:
    st.error("No data available.")
    st.stop()

# ------------------------------------------------
# MONTHLY / YEARLY DATA
# ------------------------------------------------

if selected_year == "All Years":
# Show data by year

    monthly_data = (
        df.assign(Year=df[date_col].dt.year)
        .groupby("Year", as_index=False)
        .agg(Usage=(usage_col, agg_function), Predicted=(pred_col, agg_function))
    )

    monthly_data["MonthLabel"] = monthly_data["Year"].astype(str)
    monthly_data["MonthDate"] = pd.to_datetime(
        monthly_data["Year"].astype(str) + "-01-01"
    )

else:
# Show by month for selected year

    monthly_data = (
        df.assign(Month=df[date_col].dt.to_period("M"))
        .groupby("Month", as_index=False)
        .agg(
            Usage=(usage_col, agg_function),
            Predicted=(pred_col, agg_function)
        )
    )

    monthly_data["MonthDate"] = monthly_data["Month"].dt.to_timestamp()

    monthly_data = monthly_data.sort_values("MonthDate").reset_index(drop=True)

    monthly_data["MonthLabel"] = monthly_data["MonthDate"].dt.strftime("%b %Y")

# ------------------------------------------------
# NAVIGATION
# ------------------------------------------------
st.sidebar.header("Navigation")

if selected_year == "All Years":
    selected_month_label = "Monthly Summary"
else:
    month_labels = monthly_data["MonthLabel"].tolist()

    selected_month_label = st.sidebar.selectbox(
        "Month", ["Monthly Summary"] + month_labels
    )

# =================================================
# MONTHLY VIEW
# =================================================

if selected_month_label == "Monthly Summary":

    kpi_df = df.copy()
    kpi_label = "Current Selection"

    display_kpis(kpi_df, usage_col)

    # ------------------------------------------------
    # Format Chart
    # ------------------------------------------------

    st.subheader(f"{aggregation_label} Energy Usage" f" ({selected_utility})")

    fig, ax = plt.subplots(figsize=(12, 5))
    style_chart(ax, fig)
    
    x = np.arange(len(monthly_data))
    width = 0.4

    # Determine if prediction data is visible in this view

    pred_mask = (
        monthly_data["Predicted"].notna()
        & (monthly_data["Predicted"] != 0)
        & (monthly_data["Predicted"] != monthly_data["Usage"])
    )

    actual_positions = np.where(pred_mask, x - width / 2, x)

    actual_bars = ax.bar(
        actual_positions,
        monthly_data["Usage"],
        width,
        color="steelblue",
        label="Actual",
    )

    if show_predictions and pred_mask.any():
        predicted_bars = ax.bar(
            x[pred_mask] + width / 2,
            monthly_data.loc[pred_mask, "Predicted"],
            width,
            color="skyblue",
            label="Predicted",
        )
        
        for bar in predicted_bars:
            height = bar.get_height()
            ax.annotate(
                format_short(height),
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(monthly_data["MonthLabel"], rotation=45, ha="right")

    handles, labels = ax.get_legend_handles_labels()

    if handles:
        ax.legend()

    if selected_year == "All Years":
        ax.set_title(f"{aggregation_label} Usage by Year (kWh)")
    else:
        ax.set_title(f"{aggregation_label} Usage by Month (kWh)")

    ax.set_xlabel("Month")

    ax.set_ylabel(display_column)

    ax.yaxis.set_major_formatter(formatter)

    plt.xticks(rotation=45, ha="right")

    for bar in actual_bars:
        height = bar.get_height()

        ax.annotate(
            format_short(height),
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
        ) 

    plt.tight_layout()

    st.pyplot(fig)

# =================================================
# DAILY / HOURLY DRILLDOWN
# =================================================

else:

    selected_month_date = monthly_data.loc[
        monthly_data["MonthLabel"] == selected_month_label, "MonthDate"
    ].iloc[0]

    month_data = df[
        (df[date_col].dt.year == selected_month_date.year)
        & (df[date_col].dt.month == selected_month_date.month)
    ]

    daily_data = (
        month_data.assign(Day=month_data[date_col].dt.day)
        .groupby("Day", as_index=False)
        .agg(
            Usage=(usage_col, agg_function),
            Predicted=(pred_col, agg_function)
            )
    )

    available_days = daily_data["Day"].sort_values().tolist()

    selected_day = st.sidebar.selectbox(
        "Day", ["Daily Summary"] + [str(day) for day in available_days]
    )

    # =============================================
    # DAILY VIEW
    # =============================================
    
    if selected_day == "Daily Summary":

        kpi_df = month_data.copy()
        kpi_label = selected_month_label

        display_kpis(kpi_df, usage_col)

        # ------------------------------------------------
        # Format Chart
        # ------------------------------------------------

        st.subheader(
            f"{aggregation_label} Usage by Day - "
            f"{selected_month_label} "
            f"({selected_utility})"
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        style_chart(ax, fig)
    
        x = np.arange(len(daily_data))

        actual_bars = ax.bar(
            x,
            daily_data["Usage"],
            width=0.8,
            color="steelblue",
            label="Actual",
        )

        # Determine if prediction data is visible in this view

        pred_mask = (
            daily_data["Predicted"].notna()
            & (daily_data["Predicted"] != 0)
            & (daily_data["Predicted"] != daily_data["Usage"])
        )

        if show_predictions and pred_mask.any():
            ax.plot(
                x[pred_mask],
                daily_data.loc[pred_mask, "Predicted"],
                color="skyblue",
                linewidth=3,
                marker="o",
                markersize=6,
                label="Predicted",
            )
            
        ax.set_xticks(x)
        ax.set_xticklabels(daily_data["Day"])

        handles, labels = ax.get_legend_handles_labels()

        if handles:
            ax.legend()

        ax.set_title(f"{aggregation_label} Usage by Day (kWh)")

        ax.set_xlabel("Day of Month")

        ax.set_ylabel(display_column)

        ax.yaxis.set_major_formatter(formatter)

        for bar in actual_bars:
            height = bar.get_height()

            ax.annotate(
                format_short(height),
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        st.pyplot(fig)

    # =============================================
    # HOURLY VIEW
    # =============================================
    
    else:

        selected_day = int(selected_day)

        day_data = month_data[month_data[date_col].dt.day == selected_day]

        # ------------------------------------------------
        # KPI Calculations (stats along the top)
        # ------------------------------------------------

        kpi_df = day_data.copy()

        display_kpis(kpi_df, usage_col)

        kpi_label = (
            f"{selected_month_date.strftime('%b')} "
            f"{selected_day}, "
            f"{selected_month_date.year}"
        )

        hourly_data = (
            day_data.assign(Hour=day_data[date_col].dt.hour)
            .groupby("Hour", as_index=False)
            .agg(
                Usage=(usage_col, agg_function),
                Predicted=(pred_col, agg_function)
                )
        )

        # Ensure all 24 hours are displayed even if readings are missing

        all_hours = pd.DataFrame({"Hour": list(range(24))})

        hourly_data = all_hours.merge(hourly_data, on="Hour", how="left").fillna(0)

        # ------------------------------------------------
        # Format Chart
        # ------------------------------------------------

        st.subheader(
            f"{aggregation_label} Usage - "
            f"{selected_month_date.strftime('%B')} "
            f"{selected_day}, "
            f"{selected_month_date.year} "
            f" ({selected_utility})"
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        style_chart(ax, fig)

        x = np.arange(len(hourly_data))

        actual_bars = ax.bar(
            x,
            hourly_data["Usage"],
            width=0.8,
            color="steelblue",
            label="Actual",
        )

        # Determine if prediction data is visible in this view

        pred_mask = (
            hourly_data["Predicted"].notna()
            & (hourly_data["Predicted"] != 0)
            & (hourly_data["Predicted"] != hourly_data["Usage"])
        )

        if show_predictions and pred_mask.any():
            ax.plot(
                x[pred_mask],
                hourly_data.loc[pred_mask, "Predicted"],
                color="skyblue",
                linewidth=3,
                marker="o",
                markersize=6,
                label="Predicted",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(hourly_data["Hour"])

        handles, labels = ax.get_legend_handles_labels()

        if handles:
            ax.legend()

        ax.set_title(f"{aggregation_label} Usage (kWh)")

        ax.set_xlabel("Hour of Day")

        ax.set_ylabel(display_column)

        ax.yaxis.set_major_formatter(formatter)

        for bar in actual_bars:
            height = bar.get_height()

            ax.annotate(
                format_short(height),
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        st.pyplot(fig)
