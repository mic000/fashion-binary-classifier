from assi2_DataPrep import prepare_data
import numpy as np
from sklearn.svm import SVC, LinearSVC
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, cross_val_score, cross_validate)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = prepare_data()
X_train = data["X_train_unit"]
X_test = data["X_test_unit"]
y_train_noisy = data["y_train_noisy"]
y_train_clean = data["y_train_clean"]
y_test = data["y_test"]

gamma_values = np.logspace(-2, 2, 7, base = 2)
C_values = np.logspace(-2, 2, 10, base=2)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

tuned_models = []
for gamma in gamma_values:
    c_results = []
    for val in C_values:
        model = SVC(kernel="rbf", C=float(val), gamma=float(gamma))
        validation_accuracies = cross_val_score(model, X_train, y_train_noisy, cv=skf,
                                                scoring="accuracy", n_jobs=-1)

        mean_validation_accuracy = float(validation_accuracies.mean())
        mean_validation_error = (1 - mean_validation_accuracy)
        c_results.append({"C": float(val),
                          "mean_validation_accuracy": mean_validation_accuracy,
                          "mean_validation_error": mean_validation_error,
                          "validation_error_std": float(validation_accuracies.std())})
    c_results = pd.DataFrame(c_results)

    best_index = c_results["mean_validation_error"].idxmin()
    best_C_gamma = float(c_results.loc[best_index, "C"])
    best_C_validation_error = float(c_results.loc[best_index,"mean_validation_error"])

    tuned_models.append({"gamma": float(gamma),
                         "C_gamma": best_C_gamma,
                         "C_tuning_validation_error": best_C_validation_error})

tuned_models = pd.DataFrame(tuned_models)
print(tuned_models)
# print(f"The list of gamma{gamma:.4g}, and the best of gamma is {best_C_gamma:.4g},"
#       f" and validation error is {best_C_validation_error:.4f}")


gamma_cv_results = []
for row in tuned_models:
    gamma = row["gamma"]
    C_gamma = row["C_gamma"]
    model = SVC(kernel="rbf", C=C_gamma, gamma=gamma)
    scores = cross_validate(model, X_train, y_train_noisy, cv=skf, scoring="accuracy",
                            return_train_score=True, n_jobs=-1)

    mean_train_error = float(1 - scores["train_score"].mean())
    mean_validation_error = float(1 - scores["test_score"].mean())
    gamma_cv_results.append({"gamma": gamma,
                             "C_gamma": C_gamma,
                             "mean_cv_train_error": mean_train_error,
                             "mean_cv_validation_error": mean_validation_error,
                             "validation_error_std": float(scores["test_score"].std())})

gamma_cv = pd.DataFrame(gamma_cv_results)
best_index = gamma_cv["mean_cv_validation_error"].idxmin()
best_gamma = float(gamma_cv.loc[best_index, "gamma"])
best_C = float(gamma_cv.loc[best_index, "C_gamma"])
print(f"Optimal gamma: {best_gamma:.6g}")
print(f"Corresponding C_gamma: {best_C:.6g}")



final_results = []
for row in tuned_models:
    gamma = row["gamma"]
    C_gamma = row["C_gamma"]
    model = SVC(kernel="rbf", C=C_gamma, gamma=gamma)
    model.fit(X_train, y_train_noisy)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    noisy_train_error = 1 - accuracy_score(y_train_noisy, train_predictions)
    clean_test_error = 1 - accuracy_score(y_test, test_predictions)
    final_results.append({"gamma": gamma,
                          "C_gamma": C_gamma,
                          "train_error": float(noisy_train_error),
                          "test_error": float(clean_test_error)})

final_results = pd.DataFrame(final_results)
print(final_results)



plt.figure(figsize=(8, 5))
plt.semilogx(final_results["gamma"], final_results["train_error"],
             marker="o", label="Training error (noisy labels)")
plt.semilogx(final_results["gamma"], final_results["test_error"],
             marker="o", label="Test error (clean labels)")
plt.axvline(best_gamma, linestyle="--", label=(f"The CV-selected from gamma is {best_gamma:.4g}"))
plt.xlabel("Gaussian kernel parameter gamma")
plt.ylabel("Classification error")
plt.title("RBF SVM: Training and Test Error versus Gamma")
plt.legend()
plt.grid(True)
plt.show()

#
# # linear_model = LinearSVC(C=best_linear_C, dual=False,)
# # linear_model.fit(X_train, y_train_noisy)
# # linear_test_error = 1 - linear_model.score(X_test, y_test)
#
# best_rbf_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma)
# best_rbf_model.fit(X_train, y_train_noisy)
# rbf_test_error = 1 - best_rbf_model.score(X_test, y_test)
# # print(f"Linear SVM test error: {linear_test_error:.4f}")
# print(f"RBF SVM test error: {rbf_test_error:.4f}")