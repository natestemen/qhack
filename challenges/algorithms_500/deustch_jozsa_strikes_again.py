#!/usr/bin/env python3

import sys

import pennylane as qml
from pennylane import numpy as np


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #
    dev = qml.device("default.qubit", wires=6, shots=1)

    @qml.qnode(dev)
    def circuit():
        function_number_wires = [0, 1]
        function_input_wires = [2, 3]
        function_output_wires = [4]
        help_wires = [5]

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        qml.Hadamard(wires=2)
        qml.Hadamard(wires=3)

        qml.PauliX(wires=4)
        qml.Hadamard(wires=4)

        qml.PauliX(wires=5)
        qml.Hadamard(wires=5)

        U1 = qml.transforms.get_unitary_matrix(f1, wire_order=[2, 3, 4])([2, 3, 4])
        U2 = qml.transforms.get_unitary_matrix(f2, wire_order=[2, 3, 4])([2, 3, 4])
        U3 = qml.transforms.get_unitary_matrix(f3, wire_order=[2, 3, 4])([2, 3, 4])
        U4 = qml.transforms.get_unitary_matrix(f4, wire_order=[2, 3, 4])([2, 3, 4])
        I1 = np.zeros([4, 4])
        I1[0, 0] = 1
        I2 = np.zeros([4, 4])
        I2[1, 1] = 1
        I3 = np.zeros([4, 4])
        I3[2, 2] = 1
        I4 = np.zeros([4, 4])
        I4[3, 3] = 1
        U = np.kron(I1, U1) + np.kron(I2, U2) + np.kron(I3, U3) + np.kron(I4, U4)

        qml.QubitUnitary(U, wires=[0, 1, 2, 3, 4])

        qml.Hadamard(wires=2)
        qml.Hadamard(wires=3)
        qml.PauliX(wires=2)
        qml.PauliX(wires=3)
        qml.Toffoli(wires=[2, 3, 5])

        qml.PauliX(wires=2)
        qml.PauliX(wires=3)
        qml.Hadamard(wires=2)
        qml.Hadamard(wires=3)
        qml.QubitUnitary(U, wires=[0, 1, 2, 3, 4])

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        return qml.sample(wires=[0, 1])

    sample = circuit()

    if sample[0] == 0 and sample[1] == 0:
        return "4 same"
    else:
        return "2 and 2"

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
