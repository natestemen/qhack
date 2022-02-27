#!/usr/bin/env python3

import sys

import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    theta = np.arctan(beta/alpha)
    qml.RY(2*theta,wires=0)
    qml.CNOT(wires=[0,1])

    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives y=0
        - theta_B1 (float): angle that Bob chooses when he receives y=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    theta_A = [theta_A0, theta_A1]
    theta_B = [theta_B0, theta_B1]
    
    qml.RY(-1*theta_A[x],wires=0)
    qml.RY(-1*theta_B[y],wires=1)
          

    # QHACK #

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    theta_A0=params[0]
    theta_A1=params[1]
    theta_B0=params[2]
    theta_B1=params[3]
    prob = (chsh_circuit(theta_A0,theta_A1,theta_B0,theta_B1,0, 0, alpha, beta)[0]+\
        chsh_circuit(theta_A0,theta_A1,theta_B0,theta_B1,0, 0, alpha, beta)[3]+\
            chsh_circuit(theta_A0,theta_A1,theta_B0,theta_B1,0, 1, alpha, beta)[0]+\
                chsh_circuit(theta_A0,theta_A1,theta_B0,theta_B1,0, 1, alpha, beta)[3]+\
                    chsh_circuit(theta_A0,theta_A1,theta_B0,theta_B1,1, 0, alpha, beta)[0]+\
                        chsh_circuit(theta_A0,theta_A1,theta_B0,theta_B1,1, 0, alpha, beta)[3]+\
                            chsh_circuit(theta_A0,theta_A1,theta_B0,theta_B1,1, 1, alpha, beta)[1]+\
                                chsh_circuit(theta_A0,theta_A1,theta_B0,theta_B1,1, 1, alpha, beta)[2])/4
    return prob
        

    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return 1-winning_prob(params, alpha, beta)

    # QHACK #

    #Initialize parameters, choose an optimization method and number of steps
    init_params = np.array([0,np.pi/2,np.pi/2,0])
    opt = qml.AdamOptimizer(stepsize=0.8)
    steps = 300

    # QHACK #
    
    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters 
        # QHACK #

        params = opt.step(cost, params)

        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")
