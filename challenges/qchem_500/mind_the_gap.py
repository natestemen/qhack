#!/usr/bin/env python3

import sys

import pennylane as qml
from pennylane import hf
from pennylane import numpy as np


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #

    n_qubits = len(H.wires)
    dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(angle, wires):
        qml.PauliX(0)
        qml.PauliX(1)
        qml.DoubleExcitation(angle, wires=[0, 1, 2, 3])

    @qml.qnode(dev)
    def cost_fn(angle):
        circuit(angle, wires=range(n_qubits))
        return qml.expval(H)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0)
    energy = [cost_fn(theta)]

    angles = [theta]

    max_iterations = 100
    conv_tol = 1e-06
    for _ in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))
        angles.append(theta)

        if np.abs(energy[-1] - prev_energy) <= conv_tol:
            break

    ground_state_energy = energy[-1]
    optimal_angle = angles[-1]

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def find_state(param):
        circuit(param, wires=range(n_qubits))
        return qml.state()

    ground_state = find_state(optimal_angle)

    return ground_state_energy, ground_state

    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #

    ground_state_density = np.outer(ground_state, np.conj(ground_state))

    n_qubits = len(H.wires)
    H_matrix = qml.utils.sparse_hamiltonian(H).toarray()
    return qml.Hermitian(beta * ground_state_density + H_matrix, wires=range(n_qubits))

    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #

    n_qubits = len(H.wires)
    dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(angle):
        qml.PauliX(0)
        qml.PauliX(2)
        qml.DoubleExcitation(angle, wires=[0, 1, 2, 3])

    @qml.qnode(dev)
    def cost(angle):
        circuit(angle)
        return qml.expval(H1)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0)
    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-06
    min_loss = 100

    for _ in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost, theta)

        loss = cost(theta)
        angle.append(theta)

        if loss < min_loss:
            min_loss = loss

        if np.abs(loss - prev_energy) <= conv_tol:
            break

    return cost(theta)

    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
