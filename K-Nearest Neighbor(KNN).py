import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Load the dataset
dataFrame = pd.read_csv("D:\\Microsoft VS Code\\ML\\Telco_Customer_Churn.csv")

# Drop 'customerID' column
dataFrame = dataFrame.drop(['customerID'], axis=1)

# Convert 'TotalCharges' to numeric, and handle missing values
dataFrame['TotalCharges'] = pd.to_numeric(dataFrame['TotalCharges'], errors='coerce')
dataFrame['TotalCharges'] = dataFrame['TotalCharges'].fillna(dataFrame['TotalCharges'].mean())

# Map SeniorCitizen to categorical values
dataFrame['SeniorCitizen'] = dataFrame['SeniorCitizen'].map({0: "No", 1: "Yes"})

# Encode categorical variables
for column in dataFrame.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    dataFrame[column] = le.fit_transform(dataFrame[column])

# Define features and target
X = dataFrame.drop(columns=['Churn'])
y = dataFrame['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform Grid Search to find the best value for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 31)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated accuracy:", grid_search.best_score_)

# Use the best model to make predictions
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# Evaluate the best model
accuracy_best_knn = accuracy_score(y_test, y_pred)
print("Accuracy of the best KNN model:", accuracy_best_knn)
print(classification_report(y_test, y_pred))
 