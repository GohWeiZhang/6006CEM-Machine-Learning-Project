import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataFrame = pd.read_csv("D:\\Microsoft VS Code\\ML\\insurance.csv")

# Print the first few rows of the DataFrame
print(dataFrame.head())

# Shape of the DataFrame
print("Shape of the DataFrame:", dataFrame.shape)

# Descriptive statistics of the DataFrame
print(dataFrame.describe())

# Check the data types of each column
print("Data Types of Each Column:", dataFrame.dtypes)

# Check for missing values in each column
print("\nMissing Values in Each Column:", dataFrame.isnull().sum())

# Convert 'sex', 'smoker', and 'region' columns to category type
dataFrame[['sex', 'smoker', 'region']] = dataFrame[['sex', 'smoker', 'region']].astype('category')

# Group by 'region' and sum 'charges', then sort
charges = dataFrame['charges'].groupby(dataFrame['region']).sum().sort_values(ascending=True)

# Plotting Total Charges by Region
plt.figure(figsize=(8, 6))
sns.barplot(x=charges.index, y=charges.values, palette='Blues')
plt.xlabel('Region')
plt.ylabel('Total Charges')
plt.title('Total Charges by Region')
plt.xticks(rotation=45)
plt.show()

# Plotting Average Insurance Charges by Region and Sex
plt.figure(figsize=(12, 8))
sns.barplot(x='region', y='charges', hue='sex', data=dataFrame, palette='cool')
plt.xlabel('Region')
plt.ylabel('Average Charges')
plt.title('Average Insurance Charges by Region and Sex')
plt.xticks(rotation=45)
plt.legend(title='Sex')
plt.show()

# Plotting Average Insurance Charges by Region and Smoker Status
plt.figure(figsize=(12, 8))
sns.barplot(x='region', y='charges', hue='smoker', data=dataFrame, palette='Reds_r')
plt.xlabel('Region')
plt.ylabel('Average Charges')
plt.title('Average Insurance Charges by Region and Smoker Status')
plt.xticks(rotation=45)
plt.legend(title='Smoker')
plt.show()

# Plotting Average Insurance Charges by Region and Number of Children
plt.figure(figsize=(12, 8))
sns.barplot(x='region', y='charges', hue='children', data=dataFrame, palette='Set1')
plt.xlabel('Region')
plt.ylabel('Average Charges')
plt.title('Average Insurance Charges by Region and Number of Children')
plt.xticks(rotation=45)
plt.legend(title='Number of Children')
plt.show()

# Scatter plot for Age vs Charges
sns.lmplot(x='age', y='charges', data=dataFrame, hue='smoker', palette='Set1')
plt.title('Insurance Charges vs Age')
plt.show()

# Scatter plot for BMI vs Charges
sns.lmplot(x='bmi', y='charges', data=dataFrame, hue='smoker', palette='Set2')
plt.title('Insurance Charges vs BMI')
plt.show()
