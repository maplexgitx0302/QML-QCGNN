import torch
import torch.nn as nn
import pennylane as qml
from torch_geometric.nn import MessagePassing

measurements_dict = {"X":qml.PauliX, "Y":qml.PauliY, "Z":qml.PauliZ}

# Classical Multi Layer Perceptron
class ClassicalMLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel, num_layers):
        super().__init__()
        if num_layers == 0:
            self.net = nn.Linear(in_channel, out_channel)
        else:
            net = [nn.Linear(in_channel, hidden_channel), nn.ReLU()]
            for _ in range(num_layers-2):
                net += [nn.Linear(hidden_channel, hidden_channel), nn.ReLU()]
            net += [nn.Linear(hidden_channel, out_channel)]
            self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)
    
# Element-wise classical linear
# https://stackoverflow.com/questions/51980654/pytorch-element-wise-filter-layer
class ElementwiseLinear(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(in_channel))
        self.b = nn.Parameter(torch.Tensor(in_channel))
    def forward(self, x):
        return self.w * x + self.b
    
# Quantum Multi Layer Perceptron
class QuantumMLP(nn.Module):
    def __init__(self, num_qubits, num_layers, num_reupload, measurements, device='default.qubit', diff_method="best"):
        super().__init__()
        # create a quantum MLP
        @qml.qnode(qml.device(device, wires=num_qubits), diff_method=diff_method)
        def circuit(inputs, weights):
            for i in range(num_reupload+1):
                qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights=weights[i], wires=range(num_qubits))
            return [qml.expval(measurements_dict[m[1]](wires=m[0])) for m in measurements]
        # turn the quantum circuit into a torch layer
        weight_shapes = {"weights":(num_reupload+1, num_layers, num_qubits, 3)}
        net = [qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)]
        self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)

# Quantum IQP-style
# https://arxiv.org/pdf/1804.11326.pdf
class QuantumSphericalIQP(nn.Module):
    def __init__(self, num_qubits, num_layers, num_reupload, measurements, device='default.qubit', diff_method="best"):
        super().__init__()
        # create a quantum MLP
        @qml.qnode(qml.device(device, wires=num_qubits), diff_method=diff_method)
        def circuit(inputs, weights):
            for i in range(num_reupload+1):
                # IQP encoding
                for j in range(num_qubits):
                    qml.Hadamard(wires=j)
                    if j in [0,3]:
                        qml.RZ(phi=inputs[j], wires=j)
                    elif j in [1,2,4,5]:
                        qml.CNOT(wires=[3*(j//3), j])
                        qml.RZ(phi=inputs[j], wires=j)
                        qml.CNOT(wires=[3*(j//3), j])
                qml.StronglyEntanglingLayers(weights=weights[i], wires=range(num_qubits))
            return [qml.expval(measurements_dict[m[1]](wires=m[0])) for m in measurements]
        # turn the quantum circuit into a torch layer
        weight_shapes = {"weights":(num_reupload+1, num_layers, num_qubits, 3)}
        net = [qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)]
        self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)