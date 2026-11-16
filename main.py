import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# 1. Load Dataset
# بارگذاری داده‌ها

data = load_breast_cancer()

X = data.data
y = data.target


# 2. Train-Test Split
# تقسیم داده‌ها

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Define Models
# تعریف مدل‌ها

models = {
    "DecisionTreeClassifier": DecisionTreeClassifier(
        random_state=42
    ),

    "RandomForestClassifier": RandomForestClassifier(
        random_state=42
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=1000
    )
}


# 4. Build Pipelines
# ساخت Pipeline برای هر مدل

pipelines = {}

for name, model in models.items():

    pipelines[name] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])


# 5. Model Comparison
# مقایسه مدل‌ها

results = []

for name, model_pipe in pipelines.items():

    # Cross Validation
    cv_scores = cross_val_score(
        model_pipe,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_mean = cv_scores.mean()

    # Train Model
    model_pipe.fit(X_train, y_train)

    # Prediction
    y_pred_pipe = model_pipe.predict(X_test)

    # Evaluation Metrics
    accuracy_pipe = accuracy_score(
        y_test, y_pred_pipe
    )

    precision_pipe = precision_score(
        y_test, y_pred_pipe, pos_label=0
    )

    recall_pipe = recall_score(
        y_test, y_pred_pipe, pos_label=0
    )

    f1_pipe = f1_score(
        y_test, y_pred_pipe, pos_label=0
    )

    # Save Results
    results.append({
        "name": name,
        "cv_mean": cv_mean,
        "accuracy": accuracy_pipe,
        "precision": precision_pipe,
        "recall": recall_pipe,
        "f1": f1_pipe
    })


# 6. Display Comparison Table
# نمایش جدول مقایسه

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=False))


# 7. Select Best Model
# انتخاب مدل برتر

best_model_name = results_df.loc[
    results_df["accuracy"].idxmax(), "name"
]

print("\nBest Model:", best_model_name)


# 8. Logistic Regression Threshold Analysis
# بررسی آستانه تصمیم‌گیری

logistic_pipe = pipelines["LogisticRegression"]

y_proba = logistic_pipe.predict_proba(X_test)

malignant_proba = y_proba[:, 0]

threshold = 0.3

y_pred_custom = np.where(
    malignant_proba >= threshold,
    0,
    1
)

custom_precision = precision_score(
    y_test,
    y_pred_custom,
    pos_label=0
)

custom_recall = recall_score(
    y_test,
    y_pred_custom,
    pos_label=0
)

custom_f1 = f1_score(
    y_test,
    y_pred_custom,
    pos_label=0
)


# 9. Final Logistic Regression Evaluation
# ارزیابی نهایی Logistic Regression

y_pred = logistic_pipe.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, pos_label=0
)

recall = recall_score(
    y_test, y_pred, pos_label=0
)

f1 = f1_score(
    y_test, y_pred, pos_label=0
)

confusion = confusion_matrix(y_test, y_pred)


print("\nLogistic Regression Evaluation:")

print("Accuracy:", accuracy)
print("Malignant Precision:", precision)
print("Malignant Recall:", recall)
print("Malignant F1:", f1)

print("\nConfusion Matrix:")
print(confusion)


# 10. Custom Threshold Results
# نتایج آستانه سفارشی

print("\nCustom Threshold Results:")

print("Threshold:", threshold)
print("Custom Precision:", custom_precision)
print("Custom Recall:", custom_recall)
print("Custom F1:", custom_f1)
