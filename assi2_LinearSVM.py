from assi2_DataPrep import prepare_data
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate

import pandas as pd
import matplotlib.pyplot as plt


data = prepare_data()
X_train = data["X_train"]
X_test = data["X_test"]
y_train_noisy = data["y_train_noisy"]
y_train_clean = data["y_train_clean"]
y_test = data["y_test"]

C_values = np.logspace(-2, 2, 10)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

cv_results = []
for val in C_values:
    model = LinearSVC(C=val, dual=False, random_state=123)
    scores = cross_validate(model, X_train, y_train_noisy, cv=skf, scoring="accuracy",
                            return_train_score=True, n_jobs=-1)

    cv_results.append({
        "C": float(val),
        "mean_train_accuracy": float(scores["train_score"].mean()),
        "mean_validation_accuracy": float(scores["test_score"].mean()),
        "mean_train_error": float(1 - scores["train_score"].mean()),
        "mean_validation_error": float(1 - scores["test_score"].mean()),
        "validation_error_std": float(scores["test_score"].std())
    })
cv_results = pd.DataFrame(cv_results)

best_index = cv_results["mean_validation_error"].idxmin()
best_row = cv_results.loc[best_index]

best_C = best_row["C"]
best_validation_error = best_row["mean_validation_error"]
best_validation_accuracy = best_row["mean_validation_accuracy"]
print(f"Best C at {best_C:.2g}, mean validation accuracy is {best_validation_accuracy:.4f}, "
      f"and the mean validation error is {best_validation_error:.4f}")

