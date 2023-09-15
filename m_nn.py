import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from torch_geometric.nn import MessagePassing
from functools import reduce

measurements_dict = {"I":qml.Identity, "X":qml.PauliX, "Y":qml.PauliY, "Z":qml.PauliZ}





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





# Quantum Fully Conneted Graph (disordered rotation encoding)
class QuantumDisorderedFCGraph(nn.Module):
    def __init__(self, num_idx_qubits, num_nn_qubits, num_layers, num_reupload, ctrl_enc_operator, device='default.qubit', diff_method="best"):
        super().__init__()
        self.num_idx_qubits = num_idx_qubits
        self.num_nn_qubits  = num_nn_qubits
        # prepare for constructing circuits
        num_qubits = num_idx_qubits + num_nn_qubits
        x_basis_dict = {"0":qml.Identity, "1":qml.PauliX}
        expval_measurements = []
        # constructing 2**num_idx_qubits measurements I@I@I, I@I@X, I@X@I, I@X@X, ...
        for i in range(2**num_idx_qubits):
            x_basis_bool = np.binary_repr(i)[::-1]
            x_basis_bool = "0"*(num_idx_qubits-len(x_basis_bool)) + x_basis_bool
            # add X or I measurement on each idx qubits
            xz_measurements = [x_basis_dict[x_basis_bool[j]](j) for j in range(num_idx_qubits)]
            # add a Z for non-trivial measurements on each nn qubits
            xz_measurements = [reduce(lambda x, y: x @ y, xz_measurements + [qml.PauliZ(j)]) for j in range(num_idx_qubits, num_qubits)]
            expval_measurements = expval_measurements + xz_measurements
        # constructing circuit
        @qml.qnode(qml.device(device, wires=num_qubits), diff_method=diff_method)
        def circuit(inputs, weights):
            # the inputs is flattened due to torch confusing batch and features
            inputs = inputs.reshape(-1, 3)
            # constructing controlled encoding gates
            for i in range(num_reupload+1):
                for j in range(len(inputs)):
                    control_basis  = np.binary_repr(j)[::-1]
                    control_values = "0"*(num_idx_qubits-len(control_basis)) + control_basis
                    control_values = list(map(int, control_values))
                    ctrl_enc_operator(inputs[j], control=range(num_idx_qubits), control_values=control_values)
                # add simple qml layer
                qml.StronglyEntanglingLayers(weights=weights[i], wires=range(num_idx_qubits, num_qubits))
            return [qml.expval(meas) for meas in expval_measurements]
        # turn the quantum circuit into a torch layer
        weight_shapes = {"weights":(num_reupload+1, num_layers, num_nn_qubits, 3)}
        self.net      = [qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)]
        self.net      = nn.Sequential(*self.net)
        self.circuit  = circuit
    def forward(self, x):
        x = self.net(x)
        # since the inputs is flattened, we need to unflatten them
        x = torch.unflatten(x, dim=-1, sizes=(2**self.num_idx_qubits, self.num_nn_qubits))
        # transpose last two dimensions (X,Z) -> (Z,X)
        x = x.mT
        # sum up X expvals
        x = torch.sum(x, dim=-1)
        return x