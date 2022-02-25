#!/usr/bin/env python3

import sys
import pennylane as qml
from pennylane import numpy as np
import pennylane.optimize as optimize

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
    def circuit(configs,params):
        
        for i in range(num_wires):
            qml.RY(configs[i]*params[i] + params[i+num_wires],wires=[i])
        for j in range(1,num_wires):
            qml.CNOT(wires=[j-1,j])
        qml.CNOT(wires=[j,0])
        
        for i1 in range(num_wires):
            qml.RY(configs[i1]*params[i1] + params[i1+num_wires],wires=[i1])
        for j1 in range(1,num_wires):
            qml.CNOT(wires=[j1-1,j1])
        qml.CNOT(wires=[j1,0])
        
        for i2 in range(num_wires):
            qml.RY(configs[i2]*params[i2] + params[i2+num_wires],wires=[i2])
        for j2 in range(1,num_wires):
            qml.CNOT(wires=[j2-1,j2])
        qml.CNOT(wires=[j2,0])
        
            
        return qml.probs(wires=3)
    
    # Define a cost function below with your needed arguments
    def cost(params):

        # QHACK #
        
        Y = labels
        #predictions = np.zeros(5) 
        predictions=np.array([])
        for k in range(DATA_SIZE):             
            prob = circuit(ising_configs[k],params)[0]
            
            #predictions[k] = np.around(prob*2-1)
            predictions=np.append(predictions,prob*2-1)
        # QHACK #
             

        # QHACK #

        return square_loss(Y, predictions) # DO NOT MODIFY this line
    


    init_params = np.array([1.,1.,1.,1.,0.1,0.1,0.1,0.1])
    opt = qml.AdagradOptimizer(stepsize=0.8)
    steps = 10
    params = init_params
    

    for m in range(steps):
        params = opt.step(cost, params)
        

    # QHACK #

    predictions = np.zeros(DATA_SIZE)
    for kk in range(DATA_SIZE):
         prediction = np.around(circuit(ising_configs[kk],params)[0])*2-1
         predictions[kk] = prediction.astype(int)
    return predictions
    #return accuracy(labels, predictions)


if __name__ == "__main__":
    '''inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=True
    ).reshape(DATA_SIZE, -1)'''
    inputs = np.array(
        [0,0,1,0,-1,0,0,0,0,1,0,0,1,0,-1,0,0,0,0,1,0,0,0,0,-1,1,1,0,1,-1,0,0,1,0,-1,1,0,0,1,-1,0,0,0,0,1,1,1,1,1,1,1,0,1,0,-1,0,1,0,1,-1,1,0,0,1,-1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,-1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,-1,0,1,0,1,-1,1,1,1,1,1,0,1,1,0,-1,1,0,1,1,-1,0,0,0,1,-1,0,0,1,0,-1,1,0,0,0,-1,0,0,0,0,1,0,0,0,1,-1,0,0,1,1,-1,0,0,0,0,1,1,1,1,0,-1,1,1,1,1,1,1,0,1,0,-1,1,0,0,0,-1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,-1,1,1,1,1,1,0,0,0,1,-1,1,1,1,1,1,1,1,0,1,-1,0,1,0,0,-1,1,1,1,1,1,0,0,1,1,-1,0,0,0,0,1,1,0,0,0,-1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,-1,0,0,0,0,1,1,1,0,1,-1,0,0,0,0,1,1,0,1,0,-1,1,1,0,1,-1,0,0,0,0,1,1,1,1,1,-1,1,1,1,1,1,1,0,0,1,-1,0,0,0,0,1,0,0,0,1,-1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,-1,1,0,1,1,-1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,-1,0,0,0,0,1,0,1,0,0,-1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0,0,-1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,-1,0,1,1,1,-1,0,1,1,0,-1,1,0,0,0,-1,1,1,1,1,1,0,1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,-1,0,0,0,0,1,1,0,1,0,-1,0,0,0,0,1,1,0,0,1,-1,0,1,1,1,-1,1,0,0,0,-1,0,0,0,0,1,1,1,1,1,-1,1,1,1,0,-1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,-1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,-1,1,0,0,1,-1,0,0,1,1,-1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,-1,1,1,1,1,1,0,1,0,1,-1,0,1,0,0,-1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,-1,0,0,1,0,-1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,-1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,-1,0,1,0,0,-1,1,1,1,1,1,0,0,0,0,-1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,-1,1,1,1,1,1,0,1,0,0,-1,1,1,1,1,1,0,0,1,0,-1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,-1,0,0,0,0,1,0,1,0,0,-1,1,0,1,0,-1,0,1,1,0,-1,1,1,0,0,-1,0,1,1,0,-1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,-1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,-1,0,0,0,0,1,1,0,0,1,-1,1,1,0,0,-1,0,0,0,0,-1,0,0,0,0,1,0,0,0,1,-1,1,1,1,1,1,1,0,0,1,-1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,-1,0,0,0,0,1,1,0,0,1,-1,1,0,0,0,-1,0,0,0,0,1,1,0,0,0,-1,1,0,0,1,-1,1,0,0,0,-1,1,0,0,1,-1,1,1,0,1,-1,0,0,0,0,1,0,1,0,1,-1,0,0,1,1,-1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,-1,0,1,1,1,-1,0,0,0,0,1,1,0,0,0,-1,0,1,1,0,-1,0,0,0,0,-1,0,0,0,0,1,1,1,1,1,1,1,1,0,1,-1,0,0,0,0,1,0,0,1,1,-1,1,1,1,1,-1,0,0,0,0,-1,0,0,0,0,1,1,0,0,0,-1,0,0,0,0,1,1,1,1,1,1,0,0,1,0,-1,1,0,0,0,-1,0,0,0,0,1,1,0,1,0,-1,1,0,0,1,-1,0,0,1,1,-1,1,1,0,0,-1,1,0,1,1,-1,0,0,0,0,1,1,1,1,1,1,1,1,0,1,-1,1,1,1,1,1,1,0,0,1,-1,1,0,0,0,-1,0,0,0,0,1,0,0,0,0,-1,1,0,0,1,-1,1,1,1,1,-1,0,0,0,0,-1,1,1,0,0,-1,1,1,1,1,1,1,1,1,1,-1,0,0,0,0,1,0,0,0,1,-1,0,1,1,1,-1,1,1,1,1,1,0,0,0,0,-1,1,0,1,1,-1,1,1,1,0,-1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,-1
    ], dtype=int, requires_grad=True
    ).reshape(DATA_SIZE, -1)
    
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")
    
