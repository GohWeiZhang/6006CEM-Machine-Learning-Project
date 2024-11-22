import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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

# Group by 'region' and sum 'charges', then sort with observed=True to adopt future behavior
charges = dataFrame['charges'].groupby(dataFrame['region'], observed=True).sum().sort_values(ascending=True)

# One-hot encode categorical variables
dataFrame = pd.get_dummies(dataFrame, drop_first=True)

# Convert True/False columns to 1/0
dataFrame = dataFrame.astype(int)

print(dataFrame.head())

# Define features and target variable
x = dataFrame.drop(['charges'], axis=1)
y = dataFrame['charges']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)

# Initialize and fit the Linear Regression model
Lin_reg = LinearRegression()
Lin_reg.fit(x_train, y_train)

# Print the intercept and coefficients
print("Intercept:", Lin_reg.intercept_)
print("Coefficients:", Lin_reg.coef_)

# Evaluate the model
score = Lin_reg.score(x_test, y_test)
print("Model Score (R^2):", score)

# Predictions
predictions = Lin_reg.predict(x_test)

# Calculate evaluation metrics
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Plotting predicted values vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Line of equality
plt.title("Predicted Values vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid()
plt.show()