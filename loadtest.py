import pandas as pd

print("Loading dataset...")

df = pd.read_csv("cf_ml_dataset.csv")

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Basic Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isna().sum())

print("\n--- Target Distribution (future_rating_delta_30d) ---")
print(df["future_rating_delta_30d"].describe())

print("\n--- Sample Rows ---")
print(df.sample(5))
