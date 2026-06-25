import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/dataset.csv")

print("Original Data")
print(df.head())

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numeric values with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing text values with "Unknown"
text_cols = df.select_dtypes(include=['object']).columns

for col in text_cols:
    df[col] = df[col].fillna("Unknown")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\nCleaned Data")
print(df.head())

# Save cleaned dataset
df.to_csv("reports/cleaned_dataset.csv", index=False)

# -----------------------------
# REPORTING
# -----------------------------

# Generate summary report
summary = df.describe(include='all')

summary.to_csv("reports/summary_report.csv")

print("\nSummary Report Generated!")

# -----------------------------
# VISUALIZATION
# -----------------------------

# Example chart
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

if len(numeric_cols) > 0:
    df[numeric_cols[0]].hist()

    plt.title(f"{numeric_cols[0]} Distribution")
    plt.xlabel(numeric_cols[0])
    plt.ylabel("Frequency")

    plt.savefig("charts/distribution_chart.png")

    plt.show()

print("\nChart Saved Successfully!")