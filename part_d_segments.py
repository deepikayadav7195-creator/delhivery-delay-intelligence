
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load the modelling table
df = pd.read_csv("data/modelling_table.csv")


# 1. Create a profile for each hub
hub_profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .reset_index()
    .sort_values("hub_id")
)

hub_profile.to_csv("outputs/hub_profile.csv", index=False)


# 2. Prepare the four columns for clustering
cluster_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(hub_profile[cluster_columns])


# 3. Compare separation for K = 2, 3, 4 and 5
separation_rows = []

for k in [2, 3, 4, 5]:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_data)

    within_distances = []
    across_distances = []

    for i, j in combinations(range(len(scaled_data)), 2):
        distance = np.linalg.norm(scaled_data[i] - scaled_data[j])

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    within_mean = np.mean(within_distances)
    across_mean = np.mean(across_distances)

    ratio = across_mean / within_mean if within_mean != 0 else np.nan

    sizes = pd.Series(labels).value_counts().sort_values(ascending=False)

    separation_rows.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": "-".join(map(str, sizes.tolist()))
    })

separation = pd.DataFrame(separation_rows)
separation.to_csv("outputs/part_d_separation.csv", index=False)


# 4. Run the final KMeans model with 3 clusters
final_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = final_model.fit_predict(scaled_data)


# 5. Give clusters meaningful names based on delay rate
cluster_delay = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

segment_names = {
    cluster_delay.index[0]: "Standard",
    cluster_delay.index[1]: "Watch",
    cluster_delay.index[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(segment_names)


# 6. Create the hub segment file
hub_segments = hub_profile.merge(
    df[["hub_id", "hub_city", "hub_tier"]].drop_duplicates("hub_id"),
    on="hub_id",
    how="left"
)

hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].sort_values("hub_id")

hub_segments.to_csv("outputs/hub_segments.csv", index=False)


# 7. Create a summary of each segment
segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
    .reset_index()
)

segment_profile.to_csv("outputs/segment_profile.csv", index=False)


# 8. Compare actual segments with expected tier mapping
expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}


hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(expected_segment)
)

mismatch = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

mismatch = mismatch[
    ["hub_id", "segment", "hub_city", "hub_tier", "expected_segment"]
]

mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)


print("Part D hub segmentation completed.")
print("Number of hubs:", len(hub_profile))
print("\nSegment counts:")
print(hub_segments["segment"].value_counts())
print("\nAll Part D segmentation files saved in outputs/")
