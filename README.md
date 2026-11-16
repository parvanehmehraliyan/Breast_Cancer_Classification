# Breast Cancer Classification

## Overview
This project focuses on building and evaluating machine learning models for breast cancer classification using Python and Scikit-learn.

## Dataset
The project uses the Breast Cancer Wisconsin dataset available in Scikit-learn.

## Machine Learning Models
Three classification models are implemented and compared:

- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression

## Data Preprocessing
- SimpleImputer
- StandardScaler
- Pipeline

## Model Evaluation
Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- 5-Fold Cross Validation

## Results

| Model | Accuracy | Recall | F1 Score |
|---|---:|---:|---:|
| Decision Tree | 91.23% | 92.86% | 88.64% |
| Random Forest | 95.61% | 92.86% | 93.98% |
| Logistic Regression | 98.25% | 97.62% | 97.62% |

Logistic Regression achieved the best results in this experiment.

## Technologies
- Python
- NumPy
- Pandas
- Scikit-learn

## How to Run
1. Install Python.
2. Install the required libraries:

   `pip install numpy pandas scikit-learn`

3. Run the project:

   `python breast_cancer_practice.py`

## Disclaimer
This project is for educational purposes only and is not intended for medical diagnosis.
