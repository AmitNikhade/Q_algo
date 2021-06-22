#Make sur that you've installed PyTorch
#For PyTorch latest release - Here

import pennylane as qml
from pennylane import numpy as np
import torch
from torch.autograd import Variable

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev, interface='torch')
def circuit(phi1, phi2):
 qml.RX(phi1, wires=0)
 qml.RY(phi2, wires=1)
 return qml.expval(qml.PauliZ(0))

def cost(phi1, phi2):
 expval = circuit(phi1, phi2)
 return torch.abs(expval - (-1))**2

phi = torch.tensor([0.011, 0.012], requires_grad=True)
theta = torch.tensor(0.05, requires_grad=True)

opt = torch.optim.Adam([phi, theta], lr = 0.1)

steps = 200
def closure():
    opt.zero_grad()
    loss = cost(phi, theta)
    loss.backward()
    return loss
for i in range(steps):
    opt.step(closure)