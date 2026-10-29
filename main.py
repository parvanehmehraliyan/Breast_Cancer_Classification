
# Import required libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix
)

# Load the breast cancer dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train the Random Forest model
model = RandomForestClassifier(random_state=42)

# Evaluate the model using 5-fold cross-validation
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

# Calculate the mean cross-validation accuracy
cv_mean = cv_scores.mean()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predicted labels into class names
class_names = data.target_names
predicted_names = class_names[y_pred]

# Calculate evaluation metrics for malignant cases
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
precision = precision_score(y_test, y_pred, pos_label=0)

# Calculate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Display evaluation results
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print("Precision:", precision)
print("CV Mean:", cv_mean)
print("CV Scores:", cv_scores)

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion)

# Display the first 5 predicted class names
print("Predicted classes:", predicted_names[:5])

# Display the first 5 actual class names
actual_names = class_names[y_test]
print("Actual classes:", actual_names[:5])

# Count the correct predictions
correct_count = sum(predicted_names[:5] == actual_names[:5])
print("Correct predictions:", correct_count)

# Calculate the accuracy of the first 5 predictions
accuracy_5 = (correct_count / 5) * 100
print("Accuracy of first 5 predictions:", accuracy_5, "%")