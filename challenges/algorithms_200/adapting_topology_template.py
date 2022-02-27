#! /usr/bin/python3

import sys
import pennylane as qml
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
    wire_1 = cnot.wires[0]
    wire_2 = cnot.wires[1]
    number_of_swaps = 0
    graph_ = graph
    
    while wire_2 not in graph_[wire_1]:
        for i in graph_[wire_1]:
            graph_[wire_1]=graph_[wire_1]+graph[i]
        
        number_of_swaps += 1
    return number_of_swaps * 2
        
        
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    output = n_swaps(qml.CNOT(wires=[int(i) for i in inputs]))
    print(f"{output}")
