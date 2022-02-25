#!/usr/bin/env python3

import sys

import pennylane as qml
import pennylane.optimize as optimize
from pennylane import numpy as np

DATA_SIZE = 250


def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.
    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)
    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.
    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)
    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.
    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)
    Returns:
        - predictions (list(int)): Your final model predictions
    Feel free to add any other functions than `cost` and `circuit` within the "# QHACK #" markers
    that you might need.
    """

    # QHACK #

    num_wires = ising_configs.shape[1]
    dev = qml.device("default.qubit", wires=num_wires)

    # Define a variational circuit below with your needed arguments and return something meaningful
    @qml.qnode(dev)
    def circuit(configs, params):

        angles1 = [
            configs[i] * params[i] + params[i + num_wires] for i in range(num_wires)
        ]
        angles2 = [
            configs[i] * params[i + 2*num_wires] + params[i + 3*num_wires] for i in range(num_wires)
        ]
        angles3 = [
            configs[i] * params[i + 4*num_wires] + params[i + 5*num_wires] for i in range(num_wires)
        ]
        qml.broadcast(
            qml.RY,
            wires=range(num_wires),
            parameters=angles1,
            pattern="single",
        )
        qml.broadcast(qml.CNOT, wires=range(num_wires), pattern="ring")
        qml.broadcast(
            qml.RY,
            wires=range(num_wires),
            parameters=angles2,
            pattern="single",
        )
        qml.broadcast(qml.CNOT, wires=range(num_wires), pattern="ring")
        qml.broadcast(
            qml.RY,
            wires=range(num_wires),
            parameters=angles3,
            pattern="single",
        )
        qml.broadcast(qml.CNOT, wires=range(num_wires), pattern="ring")

        return qml.probs(wires=3)

    # Define a cost function below with your needed arguments
    def cost(params):

        # QHACK #

        Y = labels
        predictions = [
            circuit(ising_configs[i], params)[0] * 2 - 1 for i in range(DATA_SIZE)
        ]

        # QHACK #

        return square_loss(Y, predictions)  # DO NOT MODIFY this line

    init_params = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]*3)
    opt = qml.AdagradOptimizer(stepsize=0.5)
    steps = 10
    params = init_params

    for _ in range(steps):
        params = opt.step(cost, params)

    predictions = [
        int(np.around(circuit(ising_configs[i], params)[0]) * 2 - 1)
        for i in range(DATA_SIZE)
    ]

    # QHACK #

    return predictions
    #return accuracy(labels, predictions)


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=True
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")
   
