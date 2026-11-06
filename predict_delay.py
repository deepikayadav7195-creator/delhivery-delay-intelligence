
import pandas as pd
import joblib

from src.shipment_prep import clean_shipments, build_features


# 1. Load the new shipments and hub information
new_shipments = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")


# 2. Load the saved model and supporting files
model = joblib.load("models/delay_model.pkl")

threshold = float(
    pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0]
)

feature_order = (
    pd.read_csv("models/feature_order.csv")["feature"].tolist()
)

impute_table = pd.read_csv("models/impute_values.csv")

medians = dict(
    zip(impute_table["column"], impute_table["median"])
)


# 3. Clean the new shipment data using saved historical medians
clean, _ = clean_shipments(
    new_shipments,
    medians=medians
)


# 4. Build the features required by the model
features = build_features(clean, hubs)

X_new = features[feature_order]


# 5. Predict delay probabilities
probabilities = model.predict_proba(X_new)[:, 1]

probabilities = pd.Series(probabilities).round(4)


# 6. Flag shipments using the saved threshold
flagged = probabilities >= threshold


# 7. Add the hub segment information
hub_segments = pd.read_csv("outputs/hub_segments.csv")

results = pd.DataFrame({
    "shipment_id": features["shipment_id"].values,
    "hub_id": features["hub_id"].values,
    "delay_probability": probabilities.values,
    "flagged": flagged.values
})

results = results.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    sort=False
)


# 8. Recommend an operational action
def recommend_action(row):
    if row["flagged"]:
        if row["segment"] == "Escalate":
            return "Escalate to hub manager"
        else:
            return "Proactive customer call"
    else:
        return "No action"


results["action"] = results.apply(
    recommend_action,
    axis=1
)


# 9. Arrange the final output columns
results = results[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]


# 10. Save the prediction results
results.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False
)


print("New shipment predictions completed.")
print("Shipments processed:", len(results))
print("Flagged shipments:", int(results["flagged"].sum()))
print("\nRecommended actions:")
print(results["action"].value_counts())
print("\nOutput saved to outputs/new_shipment_actions.csv")
