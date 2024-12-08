import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# I loaded the data from the file
# I used pandas to load the data from "ch_1/diabetes.csv".
data = pd.read_csv("ch_1/diabetes.csv")

# I replaced zero values with the median value
# I replaced the zeros in "Glucose", "BloodPressure", "SkinThickness", "Insulin", and "BMI" columns with the median value of each column.
# This makes the data more consistent.
columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for column in columns_with_zeros:
    data[column].replace(0, data[column].median(), inplace=True)

# I separated the features (X) and the target variable (y)
# "Outcome" is the target variable, and the other columns are the features.
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# I split the data into training and testing sets
# I split the data into 80% training and 20% testing. The training set is for learning, and the testing set is for evaluation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Decision Tree Classifier --- 
print("\n--- Decision Tree Classifier ---")

# I created a Decision Tree classifier and trained it with the training data.
# The model learns from the training data.
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# I made predictions on the test data
# The model predicts on the test data.
dt_predictions = decision_tree.predict(X_test)

# I evaluated the model performance
# I calculated the accuracy, precision, recall, and F1 score of the model.
dt_acc = accuracy_score(y_test, dt_predictions)  # I calculated accuracy
dt_prec = precision_score(y_test, dt_predictions)  # I calculated precision
dt_rec = recall_score(y_test, dt_predictions)  # I calculated recall
dt_f1 = f1_score(y_test, dt_predictions)  # I calculated F1 score
print(f"Accuracy: {dt_acc:.2f}, Precision: {dt_prec:.2f}, Recall: {dt_rec:.2f}, F1 Score: {dt_f1:.2f}")

# I plotted the confusion matrix for the Decision Tree
# The confusion matrix shows how well the model predicts the values. I visualized it using a heatmap.
sns.heatmap(confusion_matrix(y_test, dt_predictions), annot=True, fmt='d', cmap='Greens')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Random Forest Classifier --- 
print("\n--- Random Forest Classifier ---")

# I created a Random Forest classifier and trained it with the training data.
# Random Forest is a strong model made of many decision trees.
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# I made predictions on the test data
# I used the trained model to predict on the test data.
rf_predictions = random_forest.predict(X_test)

# I evaluated the model performance
# I calculated the accuracy, precision, recall, and F1 score of the Random Forest model.
rf_acc = accuracy_score(y_test, rf_predictions)  # I calculated accuracy
rf_prec = precision_score(y_test, rf_predictions)  # I calculated precision
rf_rec = recall_score(y_test, rf_predictions)  # I calculated recall
rf_f1 = f1_score(y_test, rf_predictions)  # I calculated F1 score
print(f"Accuracy: {rf_acc:.2f}, Precision: {rf_prec:.2f}, Recall: {rf_rec:.2f}, F1 Score: {rf_f1:.2f}")

# I plotted the confusion matrix for the Random Forest
# The confusion matrix shows how well the model predicts the values. I visualized it using a heatmap.
sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', cmap='Oranges')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Support Vector Machines (SVM) Classifier --- 
print("\n--- Support Vector Machines Classifier ---")

# I created an SVM classifier and trained it with the training data.
# SVM is a strong model used for classification problems.
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

# I made predictions on the test data
# I used the trained model to predict on the test data.
svm_predictions = svm.predict(X_test)

# I evaluated the model performance
# I calculated the accuracy, precision, recall, and F1 score of the SVM model.
svm_acc = accuracy_score(y_test, svm_predictions)  # I calculated accuracy
svm_prec = precision_score(y_test, svm_predictions)  # I calculated precision
svm_rec = recall_score(y_test, svm_predictions)  # I calculated recall
svm_f1 = f1_score(y_test, svm_predictions)  # I calculated F1 score
print(f"Accuracy: {svm_acc:.2f}, Precision: {svm_prec:.2f}, Recall: {svm_rec:.2f}, F1 Score: {svm_f1:.2f}")

# I plotted the confusion matrix for the SVM
# The confusion matrix shows how well the model predicts the values. I visualized it using a heatmap.
sns.heatmap(confusion_matrix(y_test, svm_predictions), annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
