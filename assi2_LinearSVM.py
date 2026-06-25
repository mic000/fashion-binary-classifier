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

C_values = np.logspace(-2, 2, 10, base=2)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

cv_results = []
for val in C_values:
    model = LinearSVC(C=val, dual=False, random_state=123)
    scores = cross_validate(model, X_train, y_train_noisy, cv=skf, scoring="accuracy",
                            return_train_score=True, n_jobs=-1)

    cv_results.append({"C": float(val),
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

final_results = []
for val in C_values:
    model = LinearSVC(C=val, dual=False, random_state=123)
    model.fit(X_train, y_train_noisy)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_error_noisy = 1 - accuracy_score(y_train_noisy, train_predictions)
    train_error_clean = 1 - accuracy_score(y_train_clean, train_predictions)
    test_error = 1 - accuracy_score(y_test, test_predictions)

    final_results.append({"C": float(val),
                          "train_error_noisy": float(train_error_noisy),
                          "train_error_clean": float(train_error_clean),
                          "test_error": float(test_error)
                          })
final = pd.DataFrame(final_results)



plt.figure(figsize=(8, 5))
plt.semilogx(cv_results["C"], cv_results["mean_train_error"], marker="o", label="CV training error")
plt.semilogx(cv_results["C"], cv_results["mean_validation_error"], marker="o", label="CV validation error")
plt.axvline(best_C, linestyle="--", label=f"Selected C = {best_C:.2g}")
plt.xlabel("Regularization parameter C")
plt.ylabel("Classification error")
plt.title("Linear SVM: Cross-validation Error versus C")
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8, 5))
plt.semilogx(final["C"], final["train_error_noisy"], marker="o", label="Training error of noisy labels set")
plt.semilogx(final["C"], final["test_error"], marker="o", label="Test error of clean labels")
plt.axvline(best_C, linestyle="--", label=f"CV-selected C = {best_C:.2g}")
plt.xlabel("Regularization parameter C")
plt.ylabel("Classification error")
plt.title("Linear SVM: Training and Test Error versus C")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
plt.semilogx(final["C"], final["train_error_noisy"], marker="o", label="Training error against noisy labels")
plt.semilogx(final["C"], final["train_error_clean"], marker="o", label="Training error against clean labels")
plt.semilogx( final["C"], final["test_error"], marker="o", label="Test error against clean labels")
plt.axvline(best_C, linestyle="--", label=f"CV-selected C = {best_C:.2g}")
plt.xlabel("Regularization parameter C")
plt.ylabel("Classification error")
plt.title("Effect of C under Label Noise")
plt.legend()
plt.grid(True)
plt.show()


best_model = LinearSVC(C=best_C, dual=False, random_state=123)
best_model.fit(X_train, y_train_noisy)
best_train_accuracy = best_model.score(X_train, y_train_noisy)
best_test_accuracy = best_model.score(X_test, y_test)
print(f"Selected C: {best_C},"
      f"the noisy training accuracy is {best_train_accuracy:.4f}, "
      f"and the clean test accuracy: {best_test_accuracy:.4f}"
      )