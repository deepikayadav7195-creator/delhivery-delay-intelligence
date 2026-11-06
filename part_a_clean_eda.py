import os
import numpy as np
import pandas as pd

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES
)

os.makedirs("outputs", exist_ok=True)

# Read the input files
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Basic checks before cleaning
raw_rows = len(raw)
raw_clean = raw.drop_duplicates().copy()

duplicate_rows_removed = raw_rows - len(raw_clean)
service_spellings_before = raw_clean["service_type"].nunique()

missing_distance = raw_clean["distance_km"].isna().sum()
missing_weight = raw_clean["parcel_weight_kg"].isna().sum()
missing_value = raw_clean["declared_value"].isna().sum()
missing_payment = raw_clean["payment_mode"].isna().sum()

sentinel_rows = (
    raw_clean["parcel_weight_kg"] > 100
).sum()

# Clean the shipment data
clean, medians = clean_shipments(raw, None)

service_spellings_after = clean["service_type"].nunique()

# Attach hub information and create features
modelling = build_features(clean, hubs)

# Create the target column
modelling["is_delayed"] = (
    modelling["delivery_hours"] > modelling["promised_hours"]
).astype(int)

# Save the modelling table
modelling.to_csv("data/modelling_table.csv", index=False)

# Calculate delay statistics
q1 = modelling["delivery_hours"].quantile(0.25)
q3 = modelling["delivery_hours"].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

declared = modelling["declared_value_inr"]
log_declared = np.log(declared)

checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", len(raw_clean)),
    ("missing_distance_km_before", missing_distance),
    ("missing_parcel_weight_before", missing_weight),
    ("missing_declared_value_before", missing_value),
    ("missing_payment_mode_before", missing_payment),
    ("sentinel_weight_rows", sentinel_rows),
    ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(modelling)),
    (
        "nulls_in_features_and_target",
        modelling[FEATURES + ["delivery_hours", "is_delayed"]]
        .isna().sum().sum()
    ),
    ("delayed_count", modelling["is_delayed"].sum()),
    ("delay_rate", modelling["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower_bound),
    ("hours_upper_bound", upper_bound),
    (
        "hours_outlier_count",
        (
            (modelling["delivery_hours"] < lower_bound)
            | (modelling["delivery_hours"] > upper_bound)
        ).sum()
    ),
    ("declared_value_mean_inr", declared.mean()),
    ("declared_value_median_inr", declared.median()),
    ("log_declared_value_mean", log_declared.mean()),
    ("log_declared_value_median", log_declared.median()),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])
checks_df["value"] = pd.to_numeric(
    checks_df["value"], errors="coerce"
).round(4)

checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# Prepare distance bands
modelling["distance_band"] = pd.cut(
    modelling["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True
)

# Delay evidence for service type, hub tier and distance
group_results = []

for dimension, column in [
    ("service_type", "service_type"),
    ("hub_tier", "hub_tier"),
    ("distance_band", "distance_band")
]:
    grouped = modelling.groupby(
        column, observed=True
    ).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    for _, row in grouped.iterrows():
        group_results.append({
            "dimension": dimension,
            "group": str(row[column]),
            "shipments": row["shipments"],
            "delayed": row["delayed"],
            "delay_rate": row["delay_rate"],
            "mean_hours": row["mean_hours"]
        })

delay_df = pd.DataFrame(group_results)

delay_df[["delay_rate", "mean_hours"]] = (
    delay_df[["delay_rate", "mean_hours"]].round(4)
)

delay_df.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False
)

# Correlation matrix for numeric columns
correlations = modelling.select_dtypes(
    include=np.number
).corr().round(4)

correlations.to_csv("outputs/part_a_correlations.csv")

print("Part A analysis completed.")
print("Modelling rows:", len(modelling))
print("Delayed shipments:", modelling["is_delayed"].sum())
print("Delay rate:", round(modelling["is_delayed"].mean(), 4))
print("Files saved in data/ and outputs/")