
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.shipment_prep import FEATURES, NOISE_COLUMNS


# Load the modelling dataset
df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]


# Stratified training, validation and testing split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# Function to calculate classification metrics
def get_metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        )
    }


# Check the stratified split
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))
print("Training delayed:", int(y_train.sum()))
print("Validation delayed:", int(y_val.sum()))
print("Testing delayed:", int(y_test.sum()))


# 1. Blind baseline
model_rows = []

baseline_prediction = np.zeros(len(y_val), dtype=int)

model_rows.append({
    "model": "blind_all_on_time",
    **get_metrics(y_val, baseline_prediction)
})


# 2. Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)
logistic_prediction = logistic_model.predict(X_val)

model_rows.append({
    "model": "logistic_regression",
    **get_metrics(y_val, logistic_prediction)
})


# 3. Find the best Decision Tree depth
tree_depths = [2, 3, 4, 5, 6, 8, 10]
tree_results = []

for depth in tree_depths:
    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)
    prediction = tree.predict(X_val)

    result = get_metrics(y_val, prediction)

    tree_results.append({
        "depth": depth,
        **result
    })


best_tree_result = sorted(
    tree_results,
    key=lambda row: (-row["f1"], row["depth"])
)[0]

best_depth = best_tree_result["depth"]

model_rows.append({
    "model": f"decision_tree_depth_{best_depth}",
    **{
        key: value
        for key, value in best_tree_result.items()
        if key != "depth"
    }
})


# 4. Default Random Forest
default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest.fit(X_train, y_train)
forest_prediction = default_forest.predict(X_val)

model_rows.append({
    "model": "random_forest_default",
    **get_metrics(y_val, forest_prediction)
})


pd.DataFrame(model_rows).to_csv(
    "outputs/part_c_models.csv",
    index=False
)


# 5. Feature importance and F-scores
feature_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

feature_forest.fit(X_train, y_train)

selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector_all.fit(X_train, y_train)

f_scores = selector_all.scores_
rf_importances = feature_forest.feature_importances_

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances
})

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

feature_scores["is_noise"] = (
    feature_scores["feature"].isin(NOISE_COLUMNS)
)

feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False
)


# 6. Test different numbers of selected features
selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    selectk_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    selectk_model.fit(X_train, y_train)
    prediction = selectk_model.predict(X_val)

    selectk_rows.append({
        "k": k,
        "val_f1": f1_score(
            y_val, prediction, zero_division=0
        )
    })


pd.DataFrame(selectk_rows).to_csv(
    "outputs/part_c_selectk.csv",
    index=False
)


# 7. Select seven features using training data only
selector = SelectKBest(f_classif, k=7)

X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)


# 8. Grid Search and Randomized Search
parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=parameter_grid,
    cv=5
)

grid_search.fit(X_train_selected, y_train)

grid_prediction = grid_search.best_estimator_.predict(
    X_val_selected
)

grid_val_f1 = f1_score(
    y_val, grid_prediction, zero_division=0
)


random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_selected, y_train)

random_prediction = random_search.best_estimator_.predict(
    X_val_selected
)

random_val_f1 = f1_score(
    y_val, random_prediction, zero_division=0
)


# 9. Compare tuned models with the default forest
default_7_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_7_model.fit(X_train_selected, y_train)

default_7_prediction = default_7_model.predict(
    X_val_selected
)

default_7_f1 = f1_score(
    y_val, default_7_prediction, zero_division=0
)


tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": str({
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1
        }),
        "best_cv_accuracy": np.nan,
        "val_f1": default_7_f1
    }
]

pd.DataFrame(tuning_rows).to_csv(
    "outputs/part_c_tuning.csv",
    index=False
)


# 10. Choose the best configuration
priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

best_tuning_row = sorted(
    tuning_rows,
    key=lambda row: (
        -row["val_f1"],
        priority[row["search"]]
    )
)[0]

chosen_from = best_tuning_row["search"]

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_

elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_

else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }


# 11. Fit the final model on training rows
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42,
        **chosen_params
    ))
])

final_model.fit(X_train, y_train)


# 12. Choose alert threshold using validation recall
validation_probabilities = final_model.predict_proba(X_val)[:, 1]

thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

threshold_rows = []

for threshold in thresholds:
    prediction = (
        validation_probabilities >= threshold
    ).astype(int)

    threshold_rows.append({
        "threshold": threshold,
        **get_metrics(y_val, prediction)
    })


pd.DataFrame(threshold_rows).to_csv(
    "outputs/part_c_thresholds.csv",
    index=False
)


eligible_thresholds = [
    row["threshold"]
    for row in threshold_rows
    if row["recall"] >= 0.80
]

chosen_threshold = (
    max(eligible_thresholds)
    if eligible_thresholds
    else 0.1
)


# 13. Evaluate the test data exactly once
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_prediction = (
    test_probabilities >= chosen_threshold
).astype(int)

final_test_metrics = {
    "threshold": chosen_threshold,
    **get_metrics(y_test, test_prediction)
}

pd.DataFrame([final_test_metrics]).to_csv(
    "outputs/part_c_final_test.csv",
    index=False
)


# 14. Save the final configuration for packaging
final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": (
        "None"
        if chosen_params["max_depth"] is None
        else chosen_params["max_depth"]
    ),
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)


print("\nPart C classification analysis completed.")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))
print("Best tree depth:", best_depth)
print("Selected configuration:", chosen_from)
print("Chosen threshold:", chosen_threshold)
print("Test F1:", round(final_test_metrics["f1"], 4))
print("All Part C output files saved in outputs/")
