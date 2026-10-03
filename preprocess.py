import pandas as pd

# Load dataset
df = pd.read_csv("parking_data.csv")

print("Original Dataset")
print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()

# Save cleaned dataset
df.to_csv("cleaned_parking_data.csv", index=False)

print("\nPreprocessing completed successfully!")

print("\nCleaned Dataset")
print(df.head())

print("\nDataset shape:")
print(df.shape)