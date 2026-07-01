import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt

from assi2_DataPrep import prepare_data

data = prepare_data(per_class=3000)
X_train, y_train_noisy, y_train_clean = data["X_train"],  data["y_train_noisy"], data["y_train_clean"]
X_test, y_test = data["X_test"],  data["y_test"]

p_flip = 0.2
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
nn_configs = [
    {"hidden_layer_sizes": (32,), "activation": "relu", "alpha": 1e-4},
    {"hidden_layer_sizes": (64,), "activation": "relu", "alpha": 1e-4},
    {"hidden_layer_sizes": (32,), "activation": "tanh", "alpha": 1e-4},
    {"hidden_layer_sizes": (64,), "activation": "tanh", "alpha": 1e-4},
    {"hidden_layer_sizes": (128,), "activation": "tanh", "alpha": 1e-4},
    {"hidden_layer_sizes": (32,), "activation": "logistic", "alpha": 1e-4},
    {"hidden_layer_sizes": (64,), "activation": "logistic", "alpha": 1e-4},
    {"hidden_layer_sizes": (32, 64), "activation": "relu", "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 128), "activation": "relu", "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 1e-4},
    {"hidden_layer_sizes": (128, 64), "activation": "relu", "alpha": 1e-4},
    {"hidden_layer_sizes": (32,), "activation": "relu", "alpha": 1e-2},
    {"hidden_layer_sizes": (32,), "activation": "relu", "alpha": 1e-1},
    {"hidden_layer_sizes": (32,), "activation": "relu", "alpha": 1e-0},
    {"hidden_layer_sizes": (32,), "activation": "relu", "alpha": 1e-5}]

nn_results = []
for cfg in nn_configs:
    model = MLPClassifier(solver="adam", random_state=123, **cfg)
    scores = cross_validate(model, X_train, y_train_noisy, cv=skf, scoring="accuracy", return_train_score=True,
                            n_jobs=-1)
    validation_error = 1 - scores['test_score'].mean()

    nn_results.append({**cfg, "mean_validation_accuracy": float(scores["test_score"].mean()),
                       "mean_train_error": float(1 - scores["train_score"].mean()),
                       "mean_validation_error": float(validation_error),
                       "validation_error_std": float(scores["test_score"].std())})
    print(
        f"[NN] layers={str(cfg["hidden_layer_sizes"]):<10s}, act={cfg["activation"]:<8s} α={cfg["alpha"]:<6g}  CV-err={validation_error:.4f}")

nn_results = pd.DataFrame(nn_results)
best_nn_idx = int(nn_results["mean_validation_error"].idxmin())
best_nn = nn_configs[best_nn_idx]
print(f"\n→ Best NN config = {best_nn}  CV err = {nn_results.loc[best_nn_idx, "mean_validation_error"]:.4f}")

node_grid = [8, 16, 32, 64, 128, 256, 512]
exp_nodes = []
for n in node_grid:
    model = MLPClassifier(hidden_layer_sizes=(n,), random_state=123)
    model.fit(X_train, y_train_noisy)
    exp_nodes.append({"n": n, "train_error_noisy": 1 - model.score(X_train, y_train_noisy),
                      "train_error_clean": 1 - model.score(X_train, y_train_clean),
                      "test_error": 1 - model.score(X_test, y_test)})

exp_nodes = pd.DataFrame(exp_nodes)
print(exp_nodes)

epoch_grid = [5, 10, 25, 50, 100, 200, 400]
exp_epochs = []
for it in epoch_grid:
    model = MLPClassifier(hidden_layer_sizes=best_nn["hidden_layer_sizes"],
                          activation=best_nn["activation"],
                          alpha=best_nn["alpha"], max_iter=it, random_state=123)
    model.fit(X_train, y_train_noisy)
    exp_epochs.append({"max_iter": it, "train_error_noisy": 1 - model.score(X_train, y_train_noisy),
                       "train_error_clean": 1 - model.score(X_train, y_train_clean),
                       "test_error": 1 - model.score(X_test, y_test)})

exp_epochs = pd.DataFrame(exp_epochs)
print(exp_epochs)


fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].semilogx(exp_nodes["n"], exp_nodes["train_error_noisy"], "o-", label="train error (noisy)")
ax[0].semilogx(exp_nodes["n"], exp_nodes["train_error_clean"], "o-", label="train error (clean)")
ax[0].semilogx(exp_nodes["n"], exp_nodes["test_error"], "o-", label="test error")
ax[0].axhline(p_flip, ls=":", color="grey", label=f"Noise rate p = {p_flip}")
ax[0].set(xlabel="hidden nodes", ylabel="error", title="NN: varying hidden-layer width")
ax[0].legend(); ax[0].grid(True)

ax[1].plot(exp_epochs["max_iter"], exp_epochs["train_error_noisy"], "o-", label="train error (noisy)")
ax[1].plot(exp_epochs["max_iter"], exp_epochs["train_error_clean"], "o-", label="train error (clean)")
ax[1].plot(exp_epochs["max_iter"], exp_epochs["test_error"], "o-", label="test error")
ax[1].axhline(p_flip, ls=":", color="grey", label=f"Noise rate p = {p_flip}")
ax[1].set(xlabel="max_iter", ylabel="error", title="NN: varying number of epochs")
ax[1].legend(); ax[1].grid(True)

plt.tight_layout(); plt.savefig("nn vary one.png", dpi=800, bbox_inches="tight"); plt.show()


final_nn = MLPClassifier(random_state=123, **best_nn)
final_nn.fit(X_train, y_train_noisy)
nn_train_error = 1 - final_nn.score(X_train, y_train_noisy)
nn_test_error = 1 - final_nn.score(X_test, y_test)
print(f"The best choice for tuned NN is {best_nn}, train_error={nn_train_error:.4f} and test_error={nn_test_error:.4f}")

with open("model_results.txt", "a") as f:
    f.write(f"{nn_test_error}\n")