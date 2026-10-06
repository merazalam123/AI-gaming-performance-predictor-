import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/gaming_data.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show dataset size
print("\nDataset shape:")
print(df.shape)

# Show column names
print("\nColumn names:")
print(df.columns)

# Show data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Show performance distribution
print("\nPerformance distribution:")
print(df["performance"].value_counts())