#!/usr/bin/env python3

import sys

import pennylane as qml
from pennylane import numpy as np

NUM_WIRES = 6


def triple_excitation_matrix(gamma):
    """The matrix representation of a triple-excitation Givens rotation.

    Args:
        - gamma (float): The angle of rotation

    Returns:
        - (np.ndarray): The matrix representation of a triple-excitation
    """

    # QHACK #

    dim = 2**NUM_WIRES
    matrix = np.eye(dim)

    bins = [np.binary_repr(i, width=6) for i in range(dim)]
    idx1 = bins.index("000111")
    idx2 = bins.index("111000")

    matrix[idx1, idx1] = np.cos(gamma / 2)
    matrix[idx2, idx1] = np.sin(gamma / 2)
    matrix[idx1, idx2] = -np.sin(gamma / 2)
    matrix[idx2, idx2] = np.cos(gamma / 2)

    return matrix

    # QHACK #


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit(angles):
    """Prepares the quantum state in the problem statement and returns qml.probs

    Args:
        - angles (list(float)): The relevant angles in the problem statement in this order:
        [alpha, beta, gamma]

    Returns:
        - (np.tensor): The probability of each computational basis state
    """

    # QHACK #
    alpha, beta, gamma = angles
    qml.BasisState(np.array([1, 1, 1, 0, 0, 0]), wires=[0, 1, 2, 3, 4, 5])

    qml.SingleExcitation(alpha, wires=[0, 5])
    qml.DoubleExcitation(beta, wires=[0, 1, 4, 5])
    qml.QubitUnitary(triple_excitation_matrix(gamma), wires=[0, 1, 2, 3, 4, 5])

    # QHACK #

    return qml.probs(wires=range(NUM_WIRES))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    probs = circuit(inputs).round(6)
    print(*probs, sep=",")
