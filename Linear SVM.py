import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

from assi2_DataPrep import prepare_data

data = prepare_data(per_class=3000)
X_train, y_train_noisy, y_train_clean = data["X_train"],  data["y_train_noisy"], data["y_train_clean"]
X_test, y_test = data["X_test"],  data["y_test"]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
C_grid = np.logspace(-2, 2, 10, base=10)
print("The regularization parameter C is: \n", np.round(C_grid, 3))

final_results, cv_results = [], []

for C in C_grid:
    model = LinearSVC(C=float(C), dual=False, random_state=123)
    scores = cross_validate(model, X_train, y_train_noisy, cv=skf,
                            scoring='accuracy', return_train_score=True, n_jobs=-1)

    cv_results.append({"C": float(C),
                       "mean_validation_accuracy": float(scores["test_score"].mean()),
                       "mean_train_error": float(1 - scores["train_score"].mean()),
                       "mean_validation_error": float(1 - scores["test_score"].mean()),
                       "validation_error_std": float(scores["test_score"].std())})

cv_results = pd.DataFrame(cv_results)
best_index = cv_results["mean_validation_error"].idxmin()
best = cv_results.loc[best_index]
best_C = best["C"]; best_validation_error = best["mean_validation_error"]
print(f"The best C of linear SVM at {best_C:.4g}, "
      f"and the mean validation error of the C is {best_validation_error:.4f}")

for C in C_grid:
    model = LinearSVC(C=float(C), dual=False, random_state=123)
    model.fit(X_train, y_train_noisy)
    train_error_noisy = 1 - accuracy_score(y_train_noisy, model.predict(X_train))
    train_error_clean = 1 - accuracy_score(y_train_clean, model.predict(X_train))
    test_error = 1 - accuracy_score(y_test, model.predict(X_test))

    final_results.append({"C": float(C),
                          "train_error_noisy": float(train_error_noisy),
                          "train_error_clean": float(train_error_clean),
                          "test_error": float(test_error)})

final_results = pd.DataFrame(final_results)


fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].semilogx(cv_results["C"], cv_results["mean_train_error"], marker="o", label="CV training error")
ax[0].semilogx(cv_results["C"], cv_results["mean_validation_error"], marker="o", label="CV validation error")
ax[0].axvline(best_C, linestyle='--', color='k', label=f'CV-selected C = {best_C:.2g}')
ax[0].set(xlabel='C', ylabel='classification error', title='Linear SVM — CV error vs C')
ax[0].legend(); ax[0].grid(True)

ax[1].semilogx(final_results["C"], final_results["train_error_noisy"], marker="o", label="training error of noisy")
ax[1].semilogx(final_results["C"], final_results["train_error_clean"], marker="o", label="training error against clean")
ax[1].semilogx(final_results["C"], final_results["test_error"], marker="o", label="test error of clean labels")
ax[1].axvline(best_C, ls='--', color='k', label=f'CV-selected C = {best_C:.2g}')
ax[1].set(xlabel='C', ylabel='classification error', title='Linear SVM — effect of C in full dataset')
ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.savefig('linear svm errors.png', dpi=800, bbox_inches='tight'); plt.show()


final_lin = LinearSVC(C=best_C, dual=False, random_state=123)
final_lin.fit(X_train, y_train_noisy)
best_train_accuracy = final_lin.score(X_train, y_train_noisy)
best_test_accuracy = final_lin.score(X_test, y_test)
lin_test_error = 1 - best_test_accuracy
print(f"Selected C: {best_C}, the noisy training accuracy is {best_train_accuracy:.4f}, "
      f"and the clean test accuracy: {best_test_accuracy:.4f} and the error is {lin_test_error:.4f}")

