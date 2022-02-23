#!/usr/bin/env python3

import sys

import numpy as np
import pennylane as qml
from scipy import optimize


def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #

    dev = qml.device("default.qubit", wires=6)

    @qml.qnode(dev)
    def circuit(thetas):
        t1, t2, t3 = thetas
        qml.BasisState(np.array([1, 1, 0, 0, 0, 0]), wires=range(6))
        qml.DoubleExcitation(t1, wires=[0, 1, 2, 3])
        qml.DoubleExcitation(t2, wires=[2, 3, 4, 5])
        qml.ctrl(qml.SingleExcitation, control=0)(t3, wires=[1, 3])
        return qml.state()

    dim = 2**6
    goal = np.zeros(dim)
    goal[48] = a
    goal[12] = b
    goal[3] = c
    goal[36] = d

    def error(thetas):
        return np.linalg.norm(circuit(thetas) - goal)

    thetas = optimize.minimize(error, [0, 0, 0]).x
    return thetas

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
