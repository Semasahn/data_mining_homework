# Import necessary libraries
# I am importing all the libraries I will need for the code.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler



# Load the dataset
# I am loading the data from a CSV file.
# The file path is "ch_3/auto-mpg.csv". 
file_path = "ch_3/auto-mpg.csv"  # Specify the correct file path
data = pd.read_csv(file_path)

# Clean column names
# I am cleaning the column names to remove any spaces at the beginning or end.
data.columns = data.columns.str.strip()

# Replace missing values ('?') with NaN and drop rows with NaN values
# I am replacing the '?' in the dataset with NaN and then removing the rows that contain NaN values.
data = data.replace('?', np.nan)
data = data.dropna()

# Prepare the features (X) and target (y)
# I am selecting the columns for the features (X) and target variable (y).
X = data.drop(['mpg', 'car name'], axis=1)  # Dropping 'mpg' and 'car name' columns
X = X.astype(float)  # Changing the data type to float
y = data['mpg']  # Target variable is 'mpg'

# Scale the features
# I am scaling the features to have a mean of 0 and a standard deviation of 1.
# This makes the model training easier and better.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
# I am splitting the data into training and testing sets. 80% for training, 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Linear Regression Model
# I am creating a Linear Regression model and training it with the training data.
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)  # Making predictions on the test data

# Random Forest Model
# I am creating a Random Forest model with 100 trees and training it with the training data.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)  # Making predictions on the test data

# Performance evaluation
# I am calculating the performance of both models (Linear Regression and Random Forest).
# I use MSE (Mean Squared Error), R² (R-squared), and MAE (Mean Absolute Error) to evaluate the models.
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print("\nLinear Regression Performance:")
print(f"MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}, MAE: {mae_lr:.2f}")

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("\nRandom Forest Performance:")
print(f"MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}, MAE: {mae_rf:.2f}")



# Plot Actual vs Predicted Values
# I am plotting the actual vs predicted values for both models to see how well the models performed.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color="blue", label="Linear Regression", alpha=0.7)
plt.scatter(y_test, y_pred_rf, color="orange", label="Random Forest", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)  # Diagonal line
plt.xlabel("Actual Values (MPG)")  # Label for x-axis
plt.ylabel("Predicted Values")  # Label for y-axis
plt.title("Actual vs Predicted Values")  # Title for the plot
plt.legend()  # Adding legend for both models
plt.show()  # Display the plot



# Feature Importance (Random Forest) - Horizontal Bar Chart
# I am calculating and displaying the feature importances from the Random Forest model.
# The feature importances tell us which features are the most important for predicting 'mpg'.
feature_importances = rf_model.feature_importances_
features = X.columns
print(features)  # Display the feature names
print(feature_importances)  # Display the importance scores

# I am plotting a horizontal bar chart to show the feature importances.
# This helps to understand which features are more important for the model.
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color="skyblue")  # Horizontal bar chart
plt.title("Feature Importances from Random Forest")  # Title of the plot
plt.ylabel("Features")  # Label for y-axis
plt.xlabel("Importance Score")  # Label for x-axis
plt.show()  # Display the plot
