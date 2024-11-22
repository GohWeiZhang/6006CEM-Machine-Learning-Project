import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as holdout, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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

# One-hot encode categorical variables
dataFrame = pd.get_dummies(dataFrame, drop_first=True)

# Define features and target variable
x = dataFrame.drop(['charges'], axis=1)
y = dataFrame['charges']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)

# Initialize the Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=0)

# Define a hyperparameter grid to tune
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees
    'max_depth': [None, 10, 20, 30],       # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples to split a node
    
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model with hyperparameter tuning
grid_search.fit(x_train, y_train)

# Get the best parameters from GridSearch
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model from GridSearchCV
best_rf_reg = grid_search.best_estimator_

# Evaluate the best model
score = best_rf_reg.score(x_test, y_test)
print("Tuned Model Score (R^2):", score)

# Predictions
predictions = best_rf_reg.predict(x_test)

# Calculate evaluation metrics
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Feature Importances
importances = best_rf_reg.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_rf_reg.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# If you want to list feature names instead of indices, extract them from the DataFrame
variables = x.columns

# Print the ranked feature importances
importance_list = []
for f in range(x.shape[1]):
    variable = variables[indices[f]]
    importance_list.append(variable)
    print(f"{f + 1}. {variable} ({importances[indices[f]]:.6f})")

# Plot the feature importances of the forest
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(importance_list, importances[indices], color="y", yerr=std[indices], align="center")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plotting predicted values vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Line of equality
plt.title("Predicted Values vs Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.grid()
plt.show()