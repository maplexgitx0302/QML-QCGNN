import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
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
            for _ in range(num_layers-1):
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
    def __init__(self, num_qubits, num_layers, num_reupload, measurements, device='default.qubit'):
        super().__init__()
        # create a quantum MLP
        @qml.qnode(qml.device(device, wires=num_qubits))
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
    def __init__(self, num_qubits, num_layers, num_reupload, measurements, device='default.qubit'):
        super().__init__()
        # create a quantum MLP
        @qml.qnode(qml.device(device, wires=num_qubits))
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

# Quantum Complete Graph Neural Network (QCGNN)
class QCGNN(nn.Module):
    def __init__(self, num_ir_qubits, num_nr_qubits, num_layers, num_reupload, ctrl_enc, qdevice='default.qubit', qbackend='ibmq_qasm_simulator', diff_method="best"):
        super().__init__()
        # setup quantum registers
        if "qiskit" in qdevice or "qiskit" in ctrl_enc.__name__:
            num_wk_qubits = num_ir_qubits - 1
        else:
            num_wk_qubits = 0
        self.num_ir_qubits = num_ir_qubits
        self.num_wk_qubits = num_wk_qubits
        self.num_nr_qubits = num_nr_qubits
        num_qubits = num_ir_qubits + num_wk_qubits + num_nr_qubits
        print(f"# ModelLog: Quantum device  = {qdevice} | Qubits (IR, WK, NR) = {num_ir_qubits, num_wk_qubits, num_nr_qubits}")
        if "qiskit" in qdevice and qbackend != "":
            qml_device = qml.device(qdevice, wires=num_qubits, backend=qdevice)
            print(f"# ModelLog: Quantum backend = {qbackend}")
        else:
            qml_device = qml.device(qdevice, wires=num_qubits)
        
        # setup measurement operators (IR:Combinations of {I,X}, NR:Measurements in Z)
        ir_meas_dict = {"0":qml.Identity, "1":qml.PauliX}
        expval_measurements = []
        for i in range(2**num_ir_qubits):
            # One of the combinations in IR
            ir_binary = np.binary_repr(i, width=num_ir_qubits)
            xz_measurements = [ir_meas_dict[ir_binary[q]](q) for q in range(num_ir_qubits)]
            # Measure each NR qubit in Z
            xz_measurements = [reduce(lambda x, y: x @ y, xz_measurements + [qml.PauliZ(q)]) for q in range(num_ir_qubits+num_wk_qubits, num_qubits)]
            expval_measurements = expval_measurements + xz_measurements

        # quantum circuit
        @qml.qnode(qml_device, diff_method=diff_method)
        def circuit(inputs=torch.rand(3*2**num_ir_qubits), weights=torch.rand(num_reupload+1, num_layers, num_nr_qubits, 3)):
            # the inputs is flattened due to torch confusing batch and features, so we reshape back
            inputs   = inputs.unflatten(dim=-1, sizes=(-1, 3))
            num_ptcs = inputs.shape[-2]

            # initialize the IR
            if num_ptcs == 2**num_ir_qubits:
                # i.e. number of particles = 2**num_ir_qubits
                for i in range(num_ir_qubits):
                    qml.Hadamard(wires=i)
            else:
                # i.e. number of particles < 2**num_ir_qubits
                state_vector = num_ptcs * [1/np.sqrt(num_ptcs)] + (2**num_ir_qubits-num_ptcs) * [0]
                state_vector = np.array(state_vector) / np.linalg.norm(state_vector)
                qml.QubitStateVector(state_vector, wires=range(num_ir_qubits))

            # main structure of data reupload
            for i in range(num_reupload+1):
                # data encoding with correponding control conditions
                for ir_idx in range(num_ptcs):
                    control_values = np.binary_repr(ir_idx, width=num_ir_qubits)
                    control_values = list(map(int, control_values))
                    if len(inputs.shape) > 2:
                        ctrl_enc(inputs[:, ir_idx], control_values=control_values)
                    else:
                        ctrl_enc(inputs[ir_idx], control_values=control_values)
                # parametrized gates using strongly entangling layers
                qml.StronglyEntanglingLayers(weights=weights[i], wires=range(num_ir_qubits+num_wk_qubits, num_qubits))
            return [qml.expval(meas) for meas in expval_measurements]
        self.circuit = circuit

        # use torch framework
        q_layer  = qml.qnn.TorchLayer(circuit, weight_shapes={"weights":(num_reupload+1, num_layers, num_nr_qubits, 3)})
        self.net = nn.Sequential(q_layer)
    
    def forward(self, x):
        x = self.net(x)
        # since the inputs is flattened in the circuit, we need to unflatten them
        x = torch.unflatten(x, dim=-1, sizes=(2**self.num_ir_qubits, self.num_nr_qubits))
        # transpose last two dimensions (X,Z) -> (Z,X)
        x = x.mT
        # sum up X expvals
        x = torch.sum(x, dim=-1) * (2**self.num_ir_qubits)
        return x