
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import (
    FEATURES,
    clean_shipments
)


# Create folders if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Load the modelling data
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]


# Recreate the same stratified split used in Part C
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# Read the final model configuration
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])

max_depth_value = config["max_depth"]
max_depth = (
    None
    if pd.isna(max_depth_value) or str(max_depth_value) == "None"
    else int(max_depth_value)
)

min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])


# Build the final model using the selected settings
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("model", RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    ))
])


# Train on the training rows, as in Part C
final_model.fit(X_train, y_train)


# Save the trained model
model_path = "models/delay_model.pkl"
joblib.dump(final_model, model_path)


# Save the alert threshold
pd.DataFrame({
    "threshold": [threshold]
}).to_csv("models/delay_threshold.csv", index=False)


# Save the feature order
pd.DataFrame({
    "feature": FEATURES
}).to_csv("models/feature_order.csv", index=False)


# Save historical imputation values
raw = pd.read_csv("data/raw_shipments.csv")
clean, medians = clean_shipments(raw)

pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
}).to_csv("models/impute_values.csv", index=False)


# Reload the saved model
loaded_model = joblib.load(model_path)


# Compare probabilities from the original and reloaded models
original_probabilities = final_model.predict_proba(X_test)[:, 1]
reloaded_probabilities = loaded_model.predict_proba(X_test)[:, 1]

max_difference = np.max(
    np.abs(original_probabilities - reloaded_probabilities)
)


# Save the reload verification result
reload_check = pd.DataFrame([{
    "test_rows_scored": len(X_test),
    "reload_max_abs_probability_diff": max_difference,
    "threshold": threshold
}])

reload_check.to_csv(
    "outputs/part_d_reload_check.csv",
    index=False
)


print("Model packaging completed.")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Selected features:", selector_k)
print("Alert threshold:", threshold)
print("Maximum reload probability difference:", max_difference)
print("Model and supporting files saved successfully.")
