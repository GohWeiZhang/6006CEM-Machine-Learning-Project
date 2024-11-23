import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
import matplotlib.pyplot as plt
from scipy.stats import uniform

# Load the dataset
dataFrame = pd.read_csv("D:\\Microsoft VS Code\\ML\\Telco_Customer_Churn.csv")

# Drop 'customerID' column
dataFrame = dataFrame.drop(['customerID'], axis=1)

# Convert 'TotalCharges' to numeric, and handle missing values
dataFrame['TotalCharges'] = pd.to_numeric(dataFrame['TotalCharges'], errors='coerce')
dataFrame['TotalCharges'] = dataFrame['TotalCharges'].fillna(dataFrame['TotalCharges'].mean())

# Check for missing values in each column
print("\nMissing Values in Each Column:", dataFrame.isnull().sum())

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

# Define Logistic Regression model
lr_model = LogisticRegression()

# Define the hyperparameters grid for RandomizedSearchCV
param_dist = {
    'C': uniform(0.01, 10),  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'saga'],  # Solvers for optimization
    'max_iter': [100, 200, 300],  # Maximum number of iterations
}

# Set up the RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(lr_model, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Best parameters and best score from RandomizedSearchCV
print(f"Best parameters from Random Search: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_}")

# Train the Logistic Regression model using the best hyperparameters found
best_lr_model = random_search.best_estimator_

# Predict and evaluate
y_pred = best_lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
print("Logistic Regression accuracy with Random Search is:", accuracy_lr)

# Generate ROC curve
y_pred_prob = best_lr_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot the ROC curve
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.plot(fpr, tpr, label='Logistic Regression (Random Search)', color="r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve with Random Search', fontsize=16)
plt.legend()
plt.show()

