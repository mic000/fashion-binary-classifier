from sklearn.neural_network import MLPClassifier
from assi2_DataPrep import prepare_data
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import (StratifiedKFold, cross_val_score, cross_validate)
from assi2_DataPrep import prepare_data
from sklearn.svm import SVC, LinearSVC
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = prepare_data()
X_train = data["X_train_unit"]
X_test = data["X_test_unit"]
y_train_noisy = data["y_train_noisy"]
y_train_clean = data["y_train_clean"]
y_test = data["y_test"]

nn_configs = [
    {"activation": "relu", "hidden_layer_sizes": (50,), "alpha": 0.0001},
    {"activation": "logistic", "hidden_layer_sizes": (50,), "alpha": 0.0001},
    {"activation": "tanh", "hidden_layer_sizes": (50,), "alpha": 0.0001},
    {"activation": "relu", "hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"activation": "relu", "hidden_layer_sizes": (100,), "alpha": 0.01},
    {"activation": "logistic", "hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"activation": "logistic", "hidden_layer_sizes": (100,), "alpha": 0.01},
    {"activation": "tanh", "hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"activation": "tanh", "hidden_layer_sizes": (100,), "alpha": 0.01},
    {"activation": "logistic", "hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"activation": "tanh", "hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"activation": "identity", "hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"activation": "relu", "hidden_layer_sizes": (100,50), "alpha": 0.0001},
    {"activation": "relu", "hidden_layer_sizes": (50, 25), "alpha": 0.0001}
]


nn_cv_err = []
for cfg in nn_configs:
    clf    = MLPClassifier(random_state=123, early_stop = True, **cfg)
    scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    nn_cv_err.append(1 - scores.mean())
    print(f"  [NN] layers={cfg['hidden_layer_sizes']}  act={cfg['activation']:<8s}  α={cfg['alpha']:<6g}"
          f"  CV-error={nn_cv_err[-1]:.4f}")

best_nn_cfg = nn_configs[int(np.argmin(nn_cv_err))]
print(f"  → Best NN config = {best_nn_cfg}  CV-error = {min(nn_cv_err):.4f}")