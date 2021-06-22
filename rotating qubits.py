
import pennylane as qml
from pennylane import numpy as np

device = qml.device("default.qubit", wires=2)

@qml.qnode(device)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    return qml.expval(qml.PauliZ(0))

def cost(x, y):
    return np.sin(np.abs(circuit(x, y))) - 1

opt = qml.grad(cost, argnum=[0, 1])

