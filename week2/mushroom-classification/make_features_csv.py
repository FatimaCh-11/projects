from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=73)
features = dataset.data.features.copy()

# Export 10 feature rows without the edible/poisonous answer column.
features.head(10).to_csv("mushroom_features.csv", index=False)

print("Created mushroom_features.csv")
print("Columns:", ", ".join(features.columns))