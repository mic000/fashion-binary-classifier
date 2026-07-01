import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from assi2_DataPrep import prepare_data

data = prepare_data(per_class=3000)
X_train_unit, y_train_noisy, y_train_clean = data["X_train_unit"],  data["y_train_noisy"], data["y_train_clean"]
X_test_unit, y_test = data["X_test_unit"],  data["y_test"]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
C_grid = np.logspace(-2, 2, 10, base=10)
gamma_grid = np.logspace(-4, 2, 10, base=8)
print("The regularization parameter C is: \n", np.round(C_grid, 3))
print("The gamma scale parameter is: \n", np.round(gamma_grid, 3))

cv_results = []

for G in gamma_grid:
    for C in C_grid:
        model = SVC(kernel="rbf", C=float(C), gamma=float(G))
        scores = cross_validate(model, X_train_unit, y_train_noisy, cv=skf, scoring="accuracy",
                                return_train_score=True, n_jobs=-1)

        cv_results.append({"gamma": float(G), "C": float(C),
                           "mean_validation_accuracy": float(scores["test_score"].mean()),
                           "mean_train_error": float(1 - scores["train_score"].mean()),
                           "mean_validation_error": float(1 - scores["test_score"].mean()),
                           "validation_error_std": float(scores["test_score"].std())})

cv_results = pd.DataFrame(cv_results)

tuned_per_gamma = (
    cv_results.loc[cv_results.groupby("gamma")["mean_validation_error"].idxmin()].sort_values("gamma").reset_index(
        drop=True))
print('Tuned (gamma, C_gamma) pairs: \n', tuned_per_gamma[['gamma', 'C', "mean_validation_error"]])

best_row = tuned_per_gamma.loc[tuned_per_gamma["mean_validation_error"].idxmin()]
best_gamma = float(best_row['gamma'])
best_C = float(best_row['C'])
print(
    f'Best (gamma, C) by CV: gamma = {best_gamma:.4g}, C={best_C:.4g}, '
    f'CV error = {best_row["mean_validation_error"]:.4f}')


final_results = []
for _, r in tuned_per_gamma.iterrows():
    G, C = float(r['gamma']), float(r['C'])
    model = SVC(kernel='rbf', C=C, gamma=G)
    model.fit(X_train_unit, y_train_noisy)
    train_noisy_error = 1 - accuracy_score(y_train_noisy, model.predict(X_train_unit))
    train_clean_error = 1 - accuracy_score(y_train_clean, model.predict(X_train_unit))
    test_error = 1 - accuracy_score(y_test, model.predict(X_test_unit))
    final_results.append({'gamma': G, 'C': C,
                          'clean_train_error': train_clean_error,
                          'noisy_train_error': train_noisy_error,
                          'test_error': test_error})

final_results = pd.DataFrame(final_results)

plt.figure(figsize=(9, 5))
plt.semilogx(final_results['gamma'], final_results['clean_train_error'], 'o-', label='training error in clean')
plt.semilogx(final_results['gamma'], final_results['noisy_train_error'], 'o-', label='training error in noisy')
plt.semilogx(final_results['gamma'], final_results['test_error'], 'o-', label='Test error')

for x, y, c in zip(final_results['gamma'], final_results['noisy_train_error'], final_results['C']):
    plt.annotate(f'C={c:.2g}', (x, y), textcoords='offset points',
                 xytext=(0, 10), fontsize=8, color='darkred', ha='center', fontweight='bold')

plt.axvline(best_gamma, ls='--', color='k', label=f'CV-selected gamma = {best_gamma:.3g}')
plt.xlabel('gamma')
plt.ylabel('classification error')
plt.title('RBF SVM — error vs gamma')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rbf svm errors.png', dpi=800, bbox_inches='tight');
plt.show()

final_rbf = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
final_rbf.fit(X_train_unit, y_train_noisy)
rbf_train_error = 1 - accuracy_score(y_train_noisy, final_rbf.predict(X_train_unit))
rbf_test_error = 1 - accuracy_score(y_test, final_rbf.predict(X_test_unit))
print(
    f"The best of gamma selected on RBF SVM is {best_gamma:.4g}, C is {best_C:.4g}, "
    f"train error is {rbf_train_error:.4f}, and test error is {rbf_test_error:.4f}")

with open("model_results.txt", "a") as f:
    f.write(f"{rbf_test_error}\n")