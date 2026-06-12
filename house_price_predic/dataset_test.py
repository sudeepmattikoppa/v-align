from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

# Add target column
df["Price"] = housing.target

print("First 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nStatistics:")
print(df.describe())


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

df["Price"].hist(bins=30)

plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()

import seaborn as sns

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="MedInc",
    y="Price",
    data=df
)

plt.title("Median Income vs House Price")

plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="AveRooms",
    y="Price",
    data=df
)

plt.title("Average Rooms vs House Price")

plt.show()