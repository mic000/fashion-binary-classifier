from assi2_DataPrep import prepare_data
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, cross_val_score)

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
    for C in C_values:
        model = SVC(kernel="rbf", C=float(C), gamma=float(gamma))
        validation_accuracies = cross_val_score(estimator=model,
                                                X=X_train,
                                                y=y_train_noisy,
                                                cv=skf,
                                                scoring="accuracy",
                                                n_jobs=-1)

        mean_validation_accuracy = float(validation_accuracies.mean())
        mean_validation_error = (1 - mean_validation_accuracy)
        c_results.append({"C": float(C),
                          "mean_validation_accuracy": mean_validation_accuracy,
                          "mean_validation_error": mean_validation_error,
                          "validation_error_std": float(validation_accuracies.std())
                          })

    c_results_df = pd.DataFrame(c_results)
    best_index = c_results_df["mean_validation_error"].idxmin()
    best_C_gamma = float(c_results_df.loc[best_index, "C"])
    best_C_validation_error = float(c_results_df.loc[best_index,"mean_validation_error"])
    tuned_models.append({"gamma": float(gamma),
                         "C_gamma": best_C_gamma,
                         "C_tuning_validation_error": best_C_validation_error
                         })

    print(f"The list of gamma{gamma:.4g}, and the best of gamma is {best_C_gamma:.4g},"
          f" and validation error is {best_C_validation_error:.4f}")