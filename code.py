# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('housing_data.csv')

# Data Cleaning
# Handle missing values
data = data.dropna()  # or use .fillna() for imputation

# EDA
sns.histplot(data['Price'], kde=True)
plt.title('Distribution of House Prices')

# Preprocessing
# Encode categorical features and scale numerical features if needed

# Split data into training and testing sets
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
predictions = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Feature Importance Analysis
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 8))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()
