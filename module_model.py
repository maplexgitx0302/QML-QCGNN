"""Module about components of classical and quantum models.

In this module, we define some frequently used components (submodels or
layers constructing larger models). The main model will NOT be provided
in this module, but in `main.py` (or `g_main.py`) instead.
"""

from functools import reduce
from typing import Callable

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# Pauli matrices dictionary.
PM = {"I": qml.Identity, "X": qml.PauliX,
      "Y": qml.PauliY, "Z": qml.PauliZ}


# The logging function.
def _log(message: str) -> None:
    """Printing function for log."""
    print(f"# ModelLog: {message}")


class ClassicalMLP(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            hidden_channel: int,
            num_layers: int
    ):
        """Classical MLP (feed forward neural network).

        Activation functions are default to be `nn.ReLU()`. There will 
        be NO activation functions in the last layer.

        Args:
            in_channel : int
                Dimension of initial input channels.
            out_channel : int
                Dimension of final output channels.
            hidden_channel : int
                Number of hidden neurons, assumed to be the same in each
                hidden layers.
            num_layers : int
                Number of hidden layers. If `num_layers=0`, then only an
                input layer and an output layer without activation
                functions (used for changing dimension of data only).
        """

        super().__init__()
        if num_layers == 0:
            # Note we intentionally NOT to use activation functions.
            self.net = nn.Linear(in_channel, out_channel)
        else:
            # Add an activation function for each layer.
            net = [nn.Linear(in_channel, hidden_channel), nn.ReLU()]
            for _ in range(num_layers-1):
                net += [nn.Linear(hidden_channel, hidden_channel), nn.ReLU()]
            net += [nn.Linear(hidden_channel, out_channel)]
            self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ElementwiseLinear(nn.Module):
    """Create a single connected linear layer.

    Unlike `nn.Linear` (which fully connects all neurons), this layer
    intends to perform a linear transformation for each neuron only.
    If the input has two dimensions, e.g., $x=(x_1, x_2)$, it will be 
    transformed seperately as $y_1=w_1*x_1+b_1$ and $y_2=w_2*x_2+b_2$.

    This class is modified from
    https://stackoverflow.com/questions/51980654/pytorch-element-wise-filter-layer
    """

    def __init__(self, in_channel):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(in_channel))
        self.b = nn.Parameter(torch.Tensor(in_channel))

    def forward(self, x):
        return self.w * x + self.b


class QuantumMLP(nn.Module):
    def __init__(
            self,
            num_qubits: int,
            num_layers: int,
            num_reupload: int,
            measurements: list[int, str],
            qdevice="default.qubit"
    ):
        """Quantum version MLP (feed forward QNN)

        Constructing a quantum MLP circuit to a torch layer.

        Args:
            num_qubits : int
                Number of qubits.
            num_layers : int
                Number of layers in a single strongly entangling layer.
                Equivalent to the depth of a "single" VQC ansatz.
            num_reupload : int
                Number of times that reupload the whole ansatz. Note if 
                `num_reupload=2` means the whole VQC ansatz will be 
                reuploaded (2+1) times.
            measurements : list[int, str]
                A list containing tuples with two elements. The first
                element is an integer, which corresponds to the index of\
                the qubit to be measured. The second element corresponds
                to the measurement basis, with value as a string "I", 
                "X", "Y" or "Z".
            qdevice : str ("default.qubit")
                Quantum device provided by PennyLane qml.
        """

        super().__init__()

        # Quantum circuit.
        @qml.qnode(qml.device(qdevice, wires=num_qubits))
        def circuit(inputs, weights):
            # Note the total VQC number is (num_reupload+1).
            for i in range(num_reupload+1):
                # Data embedding.
                qml.AngleEmbedding(
                    features=inputs,
                    wires=range(num_qubits), rotation='Y'
                )
                # VQC layer with parameters.
                qml.StronglyEntanglingLayers(
                    weights=weights[i],
                    wires=range(num_qubits)
                )
            # Turn measurements into observable list.
            observable_list = []
            for wires, pauli_str in measurements:
                pauli_observable = PM[pauli_str]
                observable = pauli_observable(wires=wires)
                observable_list.append(observable)
            return [qml.expval(observable) for observable in observable_list]

        # Turn the quantum circuit into a torch layer.
        weight_shapes = {"weights": (
            num_reupload+1, num_layers, num_qubits, 3)}
        self.net = nn.Sequential(qml.qnn.TorchLayer(
            circuit, weight_shapes=weight_shapes))

    def forward(self, x):
        return self.net(x)


class QCGNN_IX(nn.Module):
    def __init__(
            self,
            num_ir_qubits: int,
            num_nr_qubits: int,
            num_layers: int,
            num_reupload: int,
            ctrl_enc: Callable,
            qdevice: str = "default.qubit",
            qbackend: str = None,
            diff_method: str = "best",
            shots: int = 1024,
            scale: int = None
    ):
        """Quantum Complete Graph Neural Network (QCGNN) in {I,X}

        We first specify the quantum device (either PennyLane simulator
        or IBM quantum systems). The default setup is PennyLane's
        default quantum simulator. To use IBM quantum systems, see
        https://docs.pennylane.ai/projects/qiskit/en/latest/devices/ibmq.html
        for further detail.

        To build up the quantum circuit, we first embedding the data, 
        followed by VQC, then reupload several times if needed. 
        Eventually, we measure all combinations of IR qubits in "I" and 
        "X" basis.

        Args:
            num_ir_qubits : int
                Number of qubits in the index register (IR).
            num_nr_qubits : int
                Number of qubits in the network register (NR).
            num_layers : int
                Number of layers in a single strongly entangling layer.
                Equivalent to the depth of a "single" VQC ansatz.
            num_reupload : int
                Number of times that reupload the whole ansatz. Note if 
                `num_reupload=2` means the whole VQC ansatz will be 
                reuploaded (2+1) times.
            ctrl_enc : Callable
                The ansatz for encoding the data into the quantum
                circuit. Since the encoding methos is the important part
                of design, we write it in `main.py`.
            qdevice : str (default "default.qubit")
                Quantum device provided by PennyLane qml.
            qbackend : str (default None)
                If using IBM quantum systems, this argument corresponds
                to the backend of the real device (usually the qdevice
                will also be specified as "qiskit.ibmq").
            diff_method : str (default "best")
                The method for calculating gradients. Note in real 
                devices, usually only "parameter-shift" is allowed.
            shots : int (default 1024)
                Number of measurement shots. For PennyLane ideal 
                simulators, `shots` can be ignored since the returned
                values are ideal expectation values. For IBM quantum
                systems, `shots` is default as 1024. Note large shots
                might cause crashed when using IBM quantum systems.
            scale : int (default 2**num_ir_qubits)
                Since the QCGNN is built with aggregation function SUM,
                the output value of expectation values will be scaled
                with factor $2^{num_ir_qubits}$. Note when specifying as
                1, it is equivalent to aggregation function MEAN.
        """

        super().__init__()

        # Scale factor for measurement expectation value outputs. The default
        # value is set corresponding to the aggregation function SUM.
        self.scale = (2**num_ir_qubits) if scale is None else scale
        self.num_layers = num_layers
        self.num_reupload = num_reupload
        self.ctrl_enc = ctrl_enc

        # Initialize quantum registers (IR, NR). Note when executing on IBM
        # real devices, the multi-controlled gates need to be composed, so we
        # also need working qubits (denoted as `wk`).
        if ("qiskit" in qdevice) or ("qiskit" in qbackend):
            num_wk_qubits = num_ir_qubits - 1
        else:
            num_wk_qubits = 0
        self.num_ir_qubits = num_ir_qubits  # IR quantum register.
        self.num_wk_qubits = num_wk_qubits  # Working qubits.
        self.num_nr_qubits = num_nr_qubits  # NR quantum register.
        num_qubits = num_ir_qubits + num_wk_qubits + num_nr_qubits
        _log(f"Quantum device  = {qdevice}")
        _log(f"Quantum backend = {qbackend}")
        _log(
            f"Qubits (IR, WK, NR) = "
            f"{num_ir_qubits, num_wk_qubits, num_nr_qubits}"
        )

        # `wires` range that will be used later in quantum "gates".
        self.ir_wires = range(num_ir_qubits)
        self.nr_wires = range(num_ir_qubits+num_wk_qubits, num_qubits)

        # Create quantum device.
        if ("qiskit" in qdevice) and (qbackend is not None):
            # If real device -> specify backend and shots.
            self.qml_device = qml.device(
                qdevice, wires=num_qubits, backend=qbackend, shots=shots)
        else:
            self.qml_device = qml.device(qdevice, wires=num_qubits)
        self.diff_method = diff_method

        # Turn PennyLane quantum circuit into PyTorch layers.
        circuit = self.build_full_circuit()
        self.circuit = circuit
        weight_shapes = {"weights": (
            num_reupload+1, num_layers, num_nr_qubits, 3)}
        torch_layer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        self.net = nn.Sequential(torch_layer)

    def build_full_circuit(self) -> Callable:
        """Build up the quantum circuit."""

        @qml.qnode(self.qml_device, diff_method=self.diff_method)
        def full_circuit(inputs, weights):
            # Build up quantum gates.
            self.circuit_before_measurement(inputs, weights)

            # Get observable list.
            observables = self.observables_of_IX_combinations()

            return [qml.expval(obs_str) for obs_str in observables]

        return full_circuit

    def observables_of_IX_combinations(self) -> list:
        """Observables of {I,X} combinations

        Set measurement operators (excluding working qubits).
        IR -> Measure in all combinations of {I,X} basis.
        NR -> Measure in Z basis for each qubits individually.

        Returns:
            List: Observables of {I,X} combinations.
        """

        def bin_repr_to_observable_str(bin_repr: str):
            """Turn binary representation string to Pauli observable string"""
            observable_list = []
            for i in range(len(bin_repr)):
                # `i` also corresponds to the i-th qubit in IR.
                bit = bin_repr[i]
                if bit == "0":
                    observable = qml.Identity
                elif bit == "1":
                    observable = qml.PauliX
                observable_list.append(observable(wires=i))
            observable_str = reduce(lambda x, y: x @ y, observable_list)
            return observable_str

        # Loop over all combinations {I,X} in IR.
        observable_str_list = []
        # `dec_repr` -> decimal representations.
        for dec_repr in range(2**self.num_ir_qubits):
            # `bin_repr` -> binary representations.
            bin_repr = np.binary_repr(dec_repr, width=self.num_ir_qubits)
            # Observable string in IR.
            IR_observable_str = bin_repr_to_observable_str(bin_repr)
            for wires in self.nr_wires:
                # Observable string in NR (treat each NR qubits individually).
                NR_observable_str = qml.PauliZ(wires=wires)
                observable_str = qml.prod(IR_observable_str, NR_observable_str)
                observable_str_list.append(observable_str)
        return observable_str_list

    def circuit_before_measurement(self, inputs, weights):
        """Quantum circuit for QCGNN"""
        # The `inputs` will be automatically reshape as (batch_size, D),
        # where D is the dimension of a single flattened data. We reshape
        # the `inputs` back to correct shape, by assuming the data
        # constructed with only 3 features (pt, eta, phi). Note that in
        # `pennylane==0.31.0` above, if original inputs shape is (N, M, D),
        # it will automatically reshape to (N*M, D).
        inputs = inputs.unflatten(dim=-1, sizes=(-1, 3))

        # Now the shape becomes (batch_size, number_of_particles, 3).
        num_ptcs = inputs.shape[-2]

        # Quantum state initialization (assuming fixed number of particles).
        if num_ptcs == 2**self.num_ir_qubits:
            # Number of particles == 2**num_ir_qubits.
            qml.broadcast(qml.Hadamard, pattern="single", wires=self.ir_wires)
        else:
            # Number of particles < 2**num_ir_qubits.
            state_vector = num_ptcs * \
                [1/np.sqrt(num_ptcs)] + \
                (2**self.num_ir_qubits - num_ptcs) * [0]
            state_vector = np.array(state_vector) / \
                np.linalg.norm(state_vector)
            qml.QubitStateVector(state_vector, wires=self.ir_wires)

        # Data-reuploading (default at least once when `num_reupload` == 0).
        for re_idx in range(self.num_reupload+1):
            # Encoding data with multi-controlled gates.
            for ir_idx in range(num_ptcs):
                # `np.binary_repr` returns string.
                control_values = np.binary_repr(
                    ir_idx, width=self.num_ir_qubits)
                # `control_values` in pennylane needs list[int]
                control_values = list(map(int, control_values))
                # `pennylane==0.31.0` above handles whole batch simaltaneously.
                if len(inputs.shape) > 2:
                    # `inputs` shape == (batch, num_ptcs, 3)
                    # Note we feed in whole batch of data, but only one
                    # particle information for each data.
                    self.ctrl_enc(inputs[:, ir_idx],
                                  control_values=control_values)
                else:
                    # `inputs` shape == (num_ptcs, 3), i.e., single data
                    # Note we feed in only one particle information.
                    self.ctrl_enc(inputs[ir_idx],
                                  control_values=control_values)
            # Using strongly entangling layers for VQC ansatz.
            qml.StronglyEntanglingLayers(
                weights=weights[re_idx], wires=self.nr_wires)

    def forward(self, x):
        # Original shape of `x` is (batch, num_ptcs, 3), with 3 representing
        # features "pt", "eta" and "phi". Since PennyLane confuses with the
        # dimension of batch and features, we need to reshape `x` as
        # (batch, num_ptcs * 3), or (num_ptcs * 3) for single data only.
        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # Pass `x` through the quantum circuits, the output shape will be
        # (batch, (2**IR) * NR), where IR/NR = num_(ir/nr)_qubits respectively.
        x = self.net(x)

        # Reshape the measurement outputs to (batch, (2**IR), NR).
        x = torch.unflatten(
            x, dim=-1, sizes=(2**self.num_ir_qubits, self.num_nr_qubits))

        # Transpose to shape (batch, NR, (2**IR)).
        x = x.mT

        # Summing up (2**IR) {I,X} combinations in IR.
        x = torch.sum(x, dim=-1) * self.scale

        # `x` is now in shape (batch, NR).
        return x


class QCGNN_0(QCGNN_IX):
    def __init__(
            self,
            num_ir_qubits: int,
            num_nr_qubits: int,
            num_layers: int,
            num_reupload: int,
            ctrl_enc: Callable,
            qdevice: str = "default.qubit",
            qbackend: str = None,
            diff_method: str = "best",
            shots: int = 1024,
            scale: int = None
    ):
        """QCGNN with Hadamard transform at the last step.

        Similar to QCGNN_IX, but we do the Hadamard transform to IR in
        the last step. Also, the observable changes to measuring the 
        probability of |0>'s qubits in IR.
        """

        super().__init__(
            num_ir_qubits=num_ir_qubits,
            num_nr_qubits=num_nr_qubits,
            num_layers=num_layers,
            num_reupload=num_reupload,
            ctrl_enc=ctrl_enc,
            qdevice=qdevice,
            qbackend=qbackend,
            diff_method=diff_method,
            shots=shots,
            scale=scale
        )

    def build_full_circuit(self) -> Callable:
        """Build up the quantum circuit."""

        @qml.qnode(self.qml_device, diff_method=self.diff_method)
        def full_circuit(inputs, weights):
            """
            Returns:
                A tensor constructed by concatenating:
                    - IR_prob: Probalities of IR state.
                    - NR_expval: Expectation values of NR.
            """
            # Build up quantum gates.
            self.circuit_before_measurement(inputs, weights)
            qml.broadcast(qml.Hadamard, pattern="single", wires=self.ir_wires)

            # Get observable list.
            observables = [qml.PauliZ(wires) for wires in self.nr_wires]
            IR_prob_dim = 2 ** self.num_ir_qubits
            prob0_obs = np.zeros((IR_prob_dim, IR_prob_dim))
            prob0_obs[0][0] = 2 ** self.num_ir_qubits
            prob0_obs = qml.Hermitian(prob0_obs, wires=self.ir_wires)

            return [qml.expval(prob0_obs @ obs_str) for obs_str in observables]

        return full_circuit

    def forward(self, x):
        # Original shape of `x` is (batch, num_ptcs, 3), with 3 representing
        # features "pt", "eta" and "phi". Since PennyLane confuses with the
        # dimension of batch and features, we need to reshape `x` as
        # (batch, num_ptcs * 3), or (num_ptcs * 3) for single data only.
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = self.net(x) * self.scale

        return x
