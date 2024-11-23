import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataFrame = pd.read_csv("D:\\Microsoft VS Code\\ML\\Telco_Customer_Churn.csv")

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

# Check the unique values in the gender column (adjust the column name if needed)
print(dataFrame['gender'].value_counts())

# Clean the data by removing rows with NaN or infinite values
dataFrame_clean = dataFrame.replace([np.inf, -np.inf], np.nan).dropna()

# Drop 'customerID' column
dataFrame = dataFrame.drop(['customerID'], axis=1)

# Count the number of males and females
gender_counts = dataFrame['gender'].value_counts()

print(dataFrame['Churn'].value_counts())
churn_counts = dataFrame['Churn'].value_counts()

print(dataFrame['PaymentMethod'].value_counts())
PaymentMethod_counts = dataFrame['PaymentMethod'].value_counts()

# Churn Distribution Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=['green', 'blue'])
plt.title('Churn Distribution')
plt.show()

# Payment Method Distribution Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(PaymentMethod_counts, labels=PaymentMethod_counts.index, autopct='%1.1f%%', colors=['blue', 'pink', 'green', 'brown'])
plt.title('Payment Method Distribution')
plt.show()

# Customer Contract Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=dataFrame, x='Churn', hue='Contract', palette='Blues')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Customer Contract Distribution')
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(25, 10))
# Encode categorical variables to numeric for correlation analysis
corr = dataFrame.apply(lambda x: pd.factorize(x)[0]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, 
            annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()

# KDE Plot for Monthly Charges by Churn
sns.set_context("paper", font_scale=1.1)
plt.figure(figsize=(8, 6))
ax = sns.kdeplot(dataFrame.MonthlyCharges[dataFrame["Churn"] == 'No'], color="Red", fill=True)
ax = sns.kdeplot(dataFrame.MonthlyCharges[dataFrame["Churn"] == 'Yes'], ax=ax, color="Blue", fill=True)
ax.legend(["Not Churn", "Churn"], loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of Monthly Charges by Churn')
plt.show()
