import numpy as np
import pandas as pd

from ffnn.model import FFNN
from ffnn.utils.activation_function import Linear, ReLU, Sigmoid, Tanh, Softmax
from ffnn.utils.loss_function import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from ffnn.utils.initialization import ZeroInit, UniformInit, NormalInit
from ffnn.utils.regularizer import L1, L2

from ffnn.plot import (
    plot_training_history,
    plot_weight_distribution,
    plot_gradient_distribution
)

def train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    split = int((1 - test_ratio) * n)

    return (
        X[:split],
        X[split:],
        y[:split],
        y[split:]
    )


def main():
    # preprocessing sementara
    data = pd.read_csv("../data/datasetml_2026.csv")

    data["placement_status"] = data["placement_status"].map({
        "Placed": 1,
        "Not Placed": 0
    })

    data = pd.get_dummies(
        data,
        columns=[
            "college_tier",
            "country",
            "university_ranking_band",
            "specialization",
            "industry"
        ],
        drop_first=True
    )

    X = data.drop("placement_status", axis=1).values.astype(float)
    y = data["placement_status"].values.reshape(-1,1).astype(float)

    X_train, X_val, y_train, y_val = train_test_split(X, y)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    model = FFNN(
        layer_sizes=[X.shape[1], 16, 8, 1],
        activations=[ReLU(), ReLU(), Sigmoid()],
        loss=BinaryCrossEntropy(),
        initializer=UniformInit(-0.1, 0.1),
        regularizer=L2(0.001)
    )

    history = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        lr=0.1,
        batch_size=32,
        verbose=1
    )

    y_pred = model.predict(X_val)
    y_pred_class = (y_pred > 0.5).astype(int)

    accuracy = np.mean(y_pred_class == y_val)

    print("\nValidation Accuracy:", accuracy)

    plot_training_history(history)

    plot_weight_distribution(model.weights, layers=[0,1,2])
    plot_gradient_distribution(model.grad_w, layers=[0,1,2])


if __name__ == "__main__":
    main()