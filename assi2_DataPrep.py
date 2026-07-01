import numpy as np
import mnist_reader


def binary_classification(X, y, c0=5, c1=7):
    mask = np.isin(y, [c0, c1])
    X_binary = X[mask]
    y_binary = np.where(y[mask] == c1, 1, 0).astype(np.int64)

    return X_binary, y_binary


def prepare_data(data_path="data", c0=5, c1=7, per_class=None, p_flip=0.2, seed=123):
    rng = np.random.default_rng(seed)

    X_train, y_train = mnist_reader.load_mnist(data_path, kind="train")
    X_test, y_test = mnist_reader.load_mnist(data_path, kind="t10k")
    print("Train shape:", X_train.shape)

    X_train, y_train = binary_classification(X_train, y_train, c0=c0, c1=c1)
    X_test, y_test = binary_classification(X_test, y_test, c0=c0, c1=c1)
    print("Train shape:", X_train.shape)

    if per_class is not None:
        selected = []
        for class_label in (0, 1):
            class_indices = np.where(y_train == class_label)[0]
            sampled = rng.choice(class_indices, size=per_class, replace=False)
            selected.append(sampled)

        selected = np.concatenate(selected)
        rng.shuffle(selected)

        X_train = X_train[selected]
        y_train = y_train[selected]

    # Rescaling with dividing
    X_train = X_train / 255
    X_test = X_test / 255

    # Only rescaling for Gaussian kernel for better result
    train_norms = np.linalg.norm(X_train, axis=1, keepdims=True)
    test_norms = np.linalg.norm(X_test, axis=1, keepdims=True)
    X_train_unit = X_train / np.maximum(train_norms, 1e-12)
    X_test_unit = X_test / np.maximum(test_norms, 1e-12)

    y_train_clean = y_train.copy()
    flip_mask = rng.random(len(y_train_clean)) < p_flip
    print(f"Label noise probability is {p_flip}, "
          f"the total number of flip is {flip_mask.sum()},"
          f"and approximately rate of actual flip is {flip_mask.mean():.4f}")

    y_train_noisy = np.where(flip_mask, 1 - y_train, y_train)
    print(f"The noisy of train class: {np.bincount(y_train_noisy)}")

    return {"X_train": X_train, "X_test": X_test,
        "X_train_unit": X_train_unit, "X_test_unit": X_test_unit,
        "y_train_clean": y_train_clean, "y_train_noisy": y_train_noisy,
        "y_test": y_test, "flip_mask": flip_mask}


if __name__ == "__main__":
    prepare_data()