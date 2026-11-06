
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

from src.shipment_prep import FEATURES


# Load the modelling dataset
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["delivery_hours"]


# Split data into training, validation and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])



# Check scaling using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

checks = pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_val)},
    {"metric": "test_rows", "value": len(X_test)},
    {
        "metric": "train_scaled_max_abs_mean",
        "value": np.abs(X_train_scaled.mean(axis=0)).max()
    },
    {
        "metric": "test_scaled_max_abs_mean",
        "value": np.abs(X_test_scaled.mean(axis=0)).max()
    }
])

checks.to_csv("outputs/part_b_checks.csv", index=False)


# Function to calculate model performance
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted)
    }


metrics_rows = []


# Baseline model: predict the training average
baseline_prediction = np.full(len(y_val), y_train.mean())

baseline_metrics = calculate_metrics(y_val, baseline_prediction)

metrics_rows.append({
    "model": "Baseline",
    "split": "validation",
    **baseline_metrics
})


# Test different Decision Tree depths
tree_depths = [1, 2, 3, 4, 6, 8, 12, None]
tree_rows = []
tree_models = {}

for depth in tree_depths:
    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)
    prediction = tree.predict(X_val)

    result = calculate_metrics(y_val, prediction)
    train_prediction = tree.predict(X_train)
    train_result = calculate_metrics(y_train, train_prediction)

    tree_models[depth] = tree

    tree_rows.append({
        "max_depth": str(depth),
        "train_RMSE": train_result["RMSE"],
        **result
})


tree_results = pd.DataFrame(tree_rows)
tree_results.to_csv("outputs/part_b_tree_depth.csv", index=False)

best_depth = min(
    tree_depths,
    key=lambda depth: (
        tree_models[depth].predict(X_val).std()
        if False else calculate_metrics(
            y_val, tree_models[depth].predict(X_val)
        )["RMSE"],
        float("inf") if depth is None else depth
    )
)

best_tree = tree_models[best_depth]


# Create the main regression models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Decision Tree": best_tree,
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}


# Train and compare models on validation data
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)

    result = calculate_metrics(y_val, prediction)

    metrics_rows.append({
        "model": name,
        "split": "validation",
        **result
    })


# Compare different Gradient Boosting learning rates
learning_rates = [0.01, 0.1, 1]
gb_rows = []

for rate in learning_rates:
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )

    gb_model.fit(X_train, y_train)
    prediction = gb_model.predict(X_val)

    gb_rows.append({
        "learning_rate": rate,
        **calculate_metrics(y_val, prediction)
    })


pd.DataFrame(gb_rows).to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False
)


# Select the two best models using validation RMSE
validation_scores = {}

for name, model in models.items():
    prediction = model.predict(X_val)
    validation_scores[name] = calculate_metrics(
        y_val, prediction
    )["RMSE"]

top_two = sorted(
    validation_scores,
    key=lambda name: validation_scores[name]
)[:2]


# Cross-validation using training and validation data
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for name in top_two:
    cv_model = clone(models[name])

    scores = cross_val_score(
        cv_model,
        X_trainval,
        y_trainval,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )

    rmse_scores = -scores

    cv_rows.append({
        "model": name,
        "cv_mean_rmse": rmse_scores.mean(),
        "cv_std_rmse": rmse_scores.std()
    })


pd.DataFrame(cv_rows).to_csv(
    "outputs/part_b_cv.csv",
    index=False
)


# Select the final model using validation RMSE
best_model_name = min(
    validation_scores,
    key=lambda name: validation_scores[name]
)

final_model = clone(models[best_model_name])

# Train the selected model on training + validation data
final_model.fit(X_trainval, y_trainval)

# Evaluate the final model on the test data
test_prediction = final_model.predict(X_test)

metrics_rows.append({
    "model": best_model_name,
    "split": "test",
    **calculate_metrics(y_test, test_prediction)
})


# Save model comparison results
pd.DataFrame(metrics_rows).to_csv(
    "outputs/part_b_metrics.csv",
    index=False
)


print("Part B regression analysis completed.")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))
print("Best model:", best_model_name)
print("Best validation RMSE:", round(validation_scores[best_model_name], 4))
print("Test RMSE:", round(calculate_metrics(y_test, test_prediction)["RMSE"], 4))
print("Part B output files saved in outputs/")
