#!/usr/bin/env python3

import sys

import pennylane as qml
from pennylane import numpy as np


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.
    # oops

    # QHACK #

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():

        # QHACK #

        state = np.zeros(2**4)
        for i, theta in enumerate(thetas):
            i = 2 * i
            state[i] = np.cos(theta / 2)
            state[i + 1] = np.sin(theta / 2)

        qml.QubitStateVector(state / np.linalg.norm(state), wires=range(4))

        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
