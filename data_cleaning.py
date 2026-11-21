import pandas as pd
import numpy as np

# STEP 1: Load dataset

file_path = "dataset/RELIANCE.csv"

data = pd.read_csv(
    file_path,
    header=[0, 1],
    index_col=0
)

# Handle Yahoo Finance multi-level columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print("STOCK MARKET DATA CLEANING")
print("==========================")

# STEP 2: Display first records

print("\nFirst 10 Records:")
print(data.head(10))

# STEP 3: Dataset information

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nDataset Information:")
data.info()

# STEP 4: Check missing values

print("\nMissing Values:")
print(data.isnull().sum())

# STEP 5: Check duplicates

print("\nDuplicate Dates:")
print(data.index.duplicated().sum())

# STEP 6: Remove duplicate dates

data = data[~data.index.duplicated(keep="first")]

# STEP 7: Convert dates

data.index = pd.to_datetime(
    data.index,
    errors="coerce"
)

data = data[~data.index.isna()]

# STEP 8: Sort data

data = data.sort_index()

# STEP 9: Convert numerical columns

for column in ["Open", "High", "Low", "Close", "Volume"]:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

# STEP 10: Remove missing values

data = data.dropna(
    subset=["Open", "High", "Low", "Close", "Volume"]
)

# STEP 11: Statistical analysis

print("\nStatistical Analysis:")
print(data.describe())

# STEP 12: Save cleaned dataset

data.to_csv("dataset/RELIANCE_cleaned.csv")

print("\n==========================")
print("DATA CLEANING COMPLETED!")
print("Cleaned dataset saved successfully!")
print("==========================")

print("\nFinal Columns:")
print(data.columns.tolist())

print("First 5 Cleaned Records:")
print(data.head(5))