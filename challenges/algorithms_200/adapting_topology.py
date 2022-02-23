#!/usr/bin/env python3

import sys

import pennylane as qml
import networkx as nx
from pennylane import numpy as np

graph = {
    0: [1],
    1: [0, 2, 3, 4],
    2: [1],
    3: [1],
    4: [1, 5, 7, 8],
    5: [4, 6],
    6: [5, 7],
    7: [4, 6],
    8: [4],
}


def n_swaps(cnot):
    """Count the minimum number of swaps needed to create the equivalent CNOT.

    Args:
        - cnot (qml.Operation): A CNOT gate that needs to be implemented on the hardware
        You can find out the wires on which an operator works by asking for the 'wires' attribute: 'cnot.wires'

    Returns:
        - (int): minimum number of swaps
    """

    # QHACK #

    G = nx.Graph(graph)
    start, end = cnot.wires[0], cnot.wires[1]
    shortest_path = nx.shortest_path(G, start, end)
    if len(shortest_path) == 2:
        return 0
    return (len(nx.shortest_path(G, start, end)) - 2) * 2

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
