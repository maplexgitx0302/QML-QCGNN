"""Quantum Complete Graph Neural Network (QCGNN)

Note that there are two encoding functions below
    - pennylane_encoding: for simulation using PennyLane (such as 
      'default.qubit').
    - qiskit_encoding: The multi-qubit gates in `pennylane_encoding` are
      decomposed to single-qubit and two-qubit gates. The decomposition
      method can be found at Quantum Computation and Quantum Information
      section 4.3.

Both are equivalent but run in different quantum devices and different
quantum gate operations. For check, see `demo_compare_enc.ipynb`. 

`QCGNN_H` is provided for double check of the result of quantum simulation, 
which only differs from `QCGNN_IX` in the steps before measurement, and 
should give the same result. We will mainly use `QCGNN_IX` in this project.
"""

from typing import Optional

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Operation
import torch
import torch.nn as nn
    

def pennylane_encoding(num_ir_qubits: int, num_nr_qubits: int):
    
    def ansatz(x: torch.Tensor, control_values: list[int]):
        """Encode data on the quantum circuit.

        Args:
            x : torch.Tensor
                The input data, expected to be flattened in shape
                (batch, 3 * nr) or (3 * nr,).
            control_values : list[int]
                The control values for multi-controlled gates.
        """

        # Index register and network register.
        ni_wires = range(num_ir_qubits)
        nr_wires = range(num_ir_qubits, num_ir_qubits + num_nr_qubits)

        # Determine the rotation gate.
        num_rotation = x.shape[-1] // num_nr_qubits
        assert num_rotation in [1, 3], f"The last dimension of x should be `nr` or `3 * nr` -> x.shape = {x.shape}"

        # Unflatten the input data.
        x = x.unflatten(dim=-1, sizes=(num_nr_qubits, num_rotation))

        # Encoding data.
        if num_rotation == 3:
            for i, rotation in enumerate(['Y', 'X', 'Y']):
                ctrl = qml.ctrl(qml.AngleEmbedding, control=ni_wires, control_values=control_values)
                ctrl(features=x[..., i], wires=nr_wires, rotation=rotation)
        elif num_rotation == 1:
            ctrl = qml.ctrl(qml.AngleEmbedding, control=ni_wires, control_values=control_values)
            ctrl(features=x[..., 0], wires=nr_wires, rotation='Y')

    return ansatz


def qiskit_encoding(num_ir_qubits: int, num_nr_qubits: int):

    num_wk_qubits = num_ir_qubits - 1
    num_qubits = num_ir_qubits + num_wk_qubits + num_nr_qubits

    def control_condition_transform(control_values: list[int]):
        """Turn ctrl-0 to ctrl-1
        
        If control values == 0, use X-gate for transforming to 1.
        """

        for i, bit in enumerate(control_values):
            # `i` corresponds to the i-th qubit in IR.
            if bit == 0:
                qml.PauliX(wires=i)

    def toffoli_tranformation(inverse: bool = False):
        """Decomposition of multi-contolled gates
        
        Use Toffoli transformation for decomposition of multi-controlled gates.

        Args:
            inverse : bool
                Whether to apply inverse transformation or not.
        """

        if (not inverse) and (num_ir_qubits > 1):
            wk_qubit_t = num_ir_qubits # target qubit, also first working qubit
            qml.Toffoli(wires=(0, 1, wk_qubit_t))
        
        if inverse:
            toffoli_range = reversed(range(num_wk_qubits - 1))
        else:
            toffoli_range = range(num_wk_qubits - 1)
        
        for i in toffoli_range:
            ir_qubit_c = 2 + i # control qubit
            wk_qubit_c = num_ir_qubits + i # control qubit
            wk_qubit_t = num_ir_qubits + i + 1 # target qubit
            qml.Toffoli(wires=(ir_qubit_c, wk_qubit_c, wk_qubit_t))

        if inverse and (num_ir_qubits > 1):
            wk_qubit_t = num_ir_qubits # target qubit, also first working qubit
            qml.Toffoli(wires=(0, 1, wk_qubit_t))  

    def ansatz(x: torch.Tensor, control_values: list[int]):
        # Target wires to be encoded.
        nr_wires = range(num_qubits - num_nr_qubits, num_qubits)

        # See "Quantum Computation and Quantum Information" section 4.3.
        control_condition_transform(control_values)
        toffoli_tranformation()
        
        # The last working qubit becomes the control qubit.
        wk_qubit_c = num_ir_qubits + num_wk_qubits - 1

        # Determine the rotation gate.
        num_rotation = x.shape[-1] // num_nr_qubits
        assert num_rotation in [1, 3], f"The last dimension of x should be `nr` or `3 * nr` -> x.shape = {x.shape}"

        # Unflatten the input data.
        x = x.unflatten(dim=-1, sizes=(num_nr_qubits, num_rotation))

        # Encoding data.
        if num_rotation == 3:
            for i, rotation in enumerate(['Y', 'X', 'Y']):
                ctrl = qml.ctrl(qml.AngleEmbedding, control=wk_qubit_c, control_values=1)
                ctrl(features=x[..., i], wires=nr_wires, rotation=rotation)
        elif num_rotation == 1:
            ctrl = qml.ctrl(qml.AngleEmbedding, control=wk_qubit_c, control_values=1)
            ctrl(features=x[..., 0], wires=nr_wires, rotation='Y')
        
        toffoli_tranformation(inverse=True)
        control_condition_transform(control_values)

    return ansatz


class QCGNN_IX(nn.Module):
    def __init__(
            self,
            num_ir_qubits: int,
            num_nr_qubits: int,
            num_layers: int,
            num_reupload: int,
            num_rotation: int,
            vqc_ansatz: Operation,
            qdevice: Optional[str] = 'default.qubit',
            qbackend: Optional[str] = '',
            diff_method: Optional[str] = 'best',
            shots: Optional[int] = 1024,
            aggregation: Optional[str] = 'add',
            noise_prob: Optional[float] = 0,
            session_id: Optional[str] = None,
            tags: Optional[str] = None,
    ):
        """Quantum Complete Graph Neural Network (QCGNN) in {I,X}

        We first specify the quantum device (either PennyLane simulator
        or IBM quantum systems). The default setup is PennyLane's
        default quantum simulator. To use IBM quantum systems, see
        https://docs.pennylane.ai/projects/qiskit/en/latest/devices/ibmq.html
        for further detail.

        To build up the quantum circuit, we first embed the data, 
        followed by VQC, then reupload several times if needed. 
        Eventually, we measure all combinations of IR qubits in "I" and 
        "X" basis.

        Args:
            num_ir_qubits : int
                Number of qubits in the index register (IR).
            num_nr_qubits : int
                Number of qubits in the network register (NR).
            num_layers : int
                Number of layers in VQC.
                Equivalent to the depth of a "single" VQC ansatz.
            num_reupload : int
                Number of times of the whole VQC ansatz (and data 
                reupload), at least 1.
            num_rotation : int
                Number of rotations in the AngleEmbedding.
                If num_rotation == 1, the AngleEmbedding will be in Y.
                If num_rotation == 3, the AngleEmbedding will be in Y-X-Y.
            vqc_ansatz : Operation
                The VQC ansatz, either `qml.BasicEntanglerLayers` or
                `qml.StronglyEntanglingLayers`.
            qdevice : Optional[str] (default 'default.qubit')
                Quantum device provided by PennyLane qml.
            qbackend : Optional[str] (default '')
                If using IBM quantum systems, this argument corresponds
                to the backend of the real device (usually the qdevice
                will also be specified as 'qiskit.ibmq').
            diff_method : Optional[str] (default 'best')
                The method for calculating gradients. Note in real 
                devices, usually only "parameter-shift" is allowed.
            shots : Optional[int] (default 1024)
                Number of measurement shots. For PennyLane ideal 
                simulators, `shots` can be ignored since the returned
                values are ideal expectation values. For IBM quantum
                systems, `shots` is default as 1024. Note large shots
                might cause crashed when using IBM quantum systems.
            aggregation : Optional[str] (default 'add')
                Aggregation function ('add' or 'mean').
            noise_prob : Optional[float] (default 0)
                The probability of noise.
        """

        super().__init__()

        self.num_layers = num_layers
        self.num_reupload = num_reupload
        self.aggregation = aggregation
        self.noise_prob = noise_prob
        self.diff_method = diff_method
        self.num_rotation = num_rotation
        self.vqc_ansatz = vqc_ansatz

        # Determine the encoding method.
        if ('qiskit' in qdevice) or ('qiskit' in qbackend):
            self.encoding = qiskit_encoding(num_ir_qubits, num_nr_qubits)
        else:
            self.encoding = pennylane_encoding(num_ir_qubits, num_nr_qubits)
        
        # Initialize quantum registers (IR, NR). Note when executing on IBM
        # real devices, the multi-controlled gates need to be composed, so we
        # also need working qubits (denoted as `wk`).
        if ('qiskit' in qdevice) or ('qiskit' in qbackend):
            num_wk_qubits = num_ir_qubits - 1
        else:
            num_wk_qubits = 0
        
        self.num_ir_qubits = num_ir_qubits  # IR quantum register.
        self.num_wk_qubits = num_wk_qubits  # Working qubits.
        self.num_nr_qubits = num_nr_qubits  # NR quantum register.
        num_qubits = num_ir_qubits + num_wk_qubits + num_nr_qubits
        self.num_qubits = num_qubits

        # `wires` range that will be used later in quantum "gates".
        self.ir_wires = range(num_ir_qubits)
        self.nr_wires = range(num_ir_qubits + num_wk_qubits, num_qubits)

        # Create quantum device.
        if 'qiskit' in qdevice:
            # If real device -> specify backend and shots.
            self.qml_device = qml.device(
                qdevice, wires=num_qubits, shots=shots,
                backend=qbackend, session_id=session_id, tags=tags
            )
        elif qdevice == 'default.mixed':
            # Used for noise.
            self.qml_device = qml.device(qdevice, wires=num_qubits, shots=shots)
        else:
            # Other quantum simulators (including default.qubit).
            self.qml_device = qml.device(qdevice, wires=num_qubits)

        # Turn PennyLane quantum circuit into PyTorch layers.
        if vqc_ansatz == qml.BasicEntanglerLayers:
            weight_shapes = {'weights': (num_reupload, num_layers, num_nr_qubits)}
        elif vqc_ansatz == qml.StronglyEntanglingLayers:
            weight_shapes = {'weights': (num_reupload, num_layers, num_nr_qubits, 3)}
        torch_layer = qml.qnn.TorchLayer(self.build_full_circuit(), weight_shapes=weight_shapes)
        self.qnn = nn.Sequential(torch_layer)

    def build_full_circuit(self):
        """Build up the quantum circuit."""

        @qml.qnode(self.qml_device, diff_method=self.diff_method)
        def full_circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # In `pennylane>=0.31.0`, the inputs will be reshaped as (-1, inputs.shape[-1])
            # So the inputs are flattend from (N, P, D) to (N, P * D).
            D = self.num_nr_qubits * self.num_rotation
            inputs = inputs.unflatten(dim=-1, sizes=(-1, D))

            # Initialize the state of the quantum circuit.
            self.circuit_initialization(inputs)

            # Data reupload and VQC.
            self.circuit_evolve(inputs, weights)

            # Get observable list.
            pauli_words = self.pauli_words_of_IX_combinations()

            return [qml.expval(paruli_word) for paruli_word in pauli_words]

        return full_circuit

    def circuit_initialization(self, x: torch.Tensor):
        """Initialize the quantum circuit.

        Args:
            x : torch.Tensor
                In shape (N, P, D)
        """

        # The padded values are set to 0., so we can use `!= 0.` to filter out.
        non_mask = (x != 0.).any(dim=-1)

        if non_mask.all() == True:
            # Usually happens when no padded values.
            qml.broadcast(qml.Hadamard, pattern='single', wires=self.ir_wires)
        else:
            state_vector = non_mask.float()
            state_vector = state_vector / torch.norm(state_vector, dim=-1, keepdim=True)
            qml.QubitStateVector(state_vector, wires=self.ir_wires)

        # Noise channel 1.
        self.random_noise()

    def circuit_evolve(self, x: torch.Tensor, weights: torch.Tensor):
        """Evolve the quantum circuit.

        Args:
            x : torch.Tensor
                In shape (N, P, D)
            weights : torch.Tensor
                In shape (num_reupload, num_layers, num_nr_qubits) or
                (num_reupload, num_layers, num_nr_qubits, 3)
        """

        # Data-reuploading.
        for re_idx in range(self.num_reupload):

            # Encoding data with multi-controlled gates.
            for ir_idx, _x in enumerate(x.unbind(dim=-2)):

                # Control values depending on the index of IR.
                control_values = np.binary_repr(ir_idx, width=self.num_ir_qubits)
                control_values = list(map(int, control_values))

                # `control_values` in pennylane needs list[int]
                self.encoding(_x, control_values=control_values)

                # Noise channel 2.
                self.random_noise()

            # Using basic or strongly entangling layers for VQC ansatz.
            self.vqc_ansatz(weights=weights[re_idx], wires=self.nr_wires)
            
            # Noise channel 3.
            self.random_noise()

    def pauli_words_of_IX_combinations(self) -> list:
        """Pauli products of {I,X} combinations

        Set measurement operators (excluding working qubits).
        IR -> Measure in all combinations of {I,X} basis.
        NR -> Measure in Z basis for each qubits individually.

        Returns:
            List: Pauli strings of {I,X} combinations.
        """

        # Loop over all combinations {I,X} in IR.
        pauli_word_list = []

        # `dec_repr` -> decimal representations.
        for dec_repr in range(2 ** self.num_ir_qubits):

            # `bin_repr` -> binary representations.
            bin_repr = np.binary_repr(dec_repr, width=self.num_ir_qubits)

            # Pauli string in IR.
            ir_pauli_str = ''
            for bit in bin_repr:
                if bit == '0':
                    ir_pauli_str += 'I'
                elif bit == '1':
                    ir_pauli_str += 'X'

            # Pauli string in WR.
            wr_pauli_str = 'I' * self.num_wk_qubits

            # Pauli string in NR (treat each NR qubits individually).
            for wires in range(self.num_nr_qubits):
                nr_pauli_str = 'I' * wires
                nr_pauli_str += 'Z'
                nr_pauli_str += 'I' * (self.num_nr_qubits - wires - 1)

                pauli_str = ir_pauli_str + wr_pauli_str + nr_pauli_str
                pauli_word = qml.pauli.string_to_pauli_word(pauli_str)
                pauli_word_list.append(pauli_word)

        return pauli_word_list

    def random_noise(self):
        """Randomly apply a noise channel."""

        if self.noise_prob > 0:

            for wires in range(self.num_qubits):
                rnd_noise = np.random.randint(2)
                if rnd_noise == 0:
                    qml.DepolarizingChannel(p=self.noise_prob, wires=wires)
                elif rnd_noise == 1:
                    qml.GeneralizedAmplitudeDamping(gamma=self.noise_prob, p=0.5, wires=wires)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Feedforward the quantum circuit.

        Args:
            x : torch.Tensor
                Shape = (N, P, D).
            mask : torch.Tensor
                Shape = (N, P).
        """

        with torch.no_grad():
            if mask is None:
                non_mask = (x != 0.).any(dim=-1)
                num_ptcs = torch.sum(non_mask, axis=-1, keepdim=True)
            else:
                num_ptcs = torch.sum(~mask, axis=-1, keepdim=True)
            x = torch.flatten(x, start_dim=-2, end_dim=-1) # (N, P, D) -> (N, P * D)

        x = self.qnn(x) # (N, (2 ** n_I) * n_Q) if QCGNN_IX, (N, n_Q) if QCGNN_H

        if self.__class__.__name__ == 'QCGNN_IX':
            sizes = (2 ** self.num_ir_qubits, self.num_nr_qubits)
            x = torch.unflatten(x, dim=-1, sizes=sizes) # (N, (2 ** n_I), n_Q)
            x = x.mT # (N, n_Q, (2 ** n_I))
            x = torch.sum(x, dim=-1) # (N, n_Q)

        # Additional aggregation from graph aggregation.
        if self.aggregation == 'add':
            x = x * num_ptcs # (N, n_Q)
        elif self.aggregation == 'mean':
            x = x / num_ptcs # (N, n_Q)

        return x


class QCGNN_H(QCGNN_IX):
    """QCGNN with Hadamard transform at the last step.

        Similar to QCGNN_IX, but we do the Hadamard transform to IR in
        the last step. Also, the observable changes to measuring the 
        probability of |0>'s qubits in IR.
    """

    def build_full_circuit(self):
        """Build up the quantum circuit."""

        @qml.qnode(self.qml_device, diff_method=self.diff_method)
        def full_circuit(inputs: torch.Tensor, weights: torch.Tensor):
            """
            Returns:
                A tensor constructed by concatenating:
                - IR_prob: Probalities of IR state.
                - NR_expval: Expectation values of NR.
            """
            # Build up quantum gates.
            D = self.num_nr_qubits * self.num_rotation
            inputs = inputs.unflatten(dim=-1, sizes=(-1, D))
            self.circuit_initialization(inputs)
            self.circuit_evolve(inputs, weights)

            # Different from QCGNN_IX, we add Hadamard transform to IR.
            qml.broadcast(qml.Hadamard, pattern='single', wires=self.ir_wires)

            # Get observable list.
            observables = [qml.PauliZ(wires) for wires in self.nr_wires]
            IR_prob_dim = 2 ** self.num_ir_qubits
            prob0_obs = np.zeros((IR_prob_dim, IR_prob_dim))
            prob0_obs[0][0] = 2 ** self.num_ir_qubits
            prob0_obs = qml.Hermitian(prob0_obs, wires=self.ir_wires)

            return [qml.expval(prob0_obs @ obs_str) for obs_str in observables]

        return full_circuit


class QuantumRotQCGNN(nn.Module):
    def __init__(
            self,
            num_ir_qubits: int,
            num_nr_qubits: int,
            num_layers: int,
            num_reupload: int,
            vqc_ansatz: Operation,
            score_dim: int,
            dropout: Optional[float] = 0.0,
            **kwargs
    ):
        """This quantum model is based on the `QCGNN_IX`.

        Args:
            score_dim : int
                Dimension of the final score output.
            dropout : float (default=0.0)
                Dropout rate for MLP hidden layers.
        """
        
        super().__init__()

        self.num_nr_qubits = num_nr_qubits

        self.phi = QCGNN_IX(
            num_ir_qubits=num_ir_qubits,
            num_nr_qubits=num_nr_qubits,
            num_layers=num_layers,
            num_reupload=num_reupload,
            num_rotation=3,
            vqc_ansatz=vqc_ansatz,
            **kwargs
        )

        # Output is in shape (batch, NR).
        self.mlp = nn.Sequential(
            nn.Linear(num_nr_qubits, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, score_dim),
        )
    
    def forward(self, x: torch.Tensor):
        """
            x : torch.Tensor
                Shape = (N, P, 3).
        """

        with torch.no_grad():
            mask = torch.isnan(x[..., 0]) # (N, P)
            x = x.masked_fill(torch.isnan(x), 0.)
            x = x.unsqueeze(-2) # (N, P, 1, 3)
            x = x.expand(-1, -1, self.num_nr_qubits, -1) # (N, P, n_Q, 3)
            x = x.flatten(start_dim=-2, end_dim=-1) # (N, P, n_Q * 3)

        x = self.phi(x, mask) # (N, n_Q)
        x = self.mlp(x)

        return x
    

class HybridQCGNN(nn.Module):
    def __init__(
            self,
            num_ir_qubits: int,
            num_nr_qubits: int,
            num_layers: int,
            num_reupload: int,
            num_rotation: int,
            vqc_ansatz: Operation,
            score_dim: int,
            dropout: Optional[float] = 0.0,
            **kwargs
    ):
        """This quantum model is based on the `QCGNN_RY`.

        Args:
            score_dim : int
                Dimension of the final score output.
            dropout : float (default=0.0)
                Dropout rate for MLP hidden layers.
        """
        
        super().__init__()
        
        # Transform (pt, eta, phi) to higher dimension feature space.
        self.embed = nn.Linear(3, num_rotation * num_nr_qubits)

        self.phi = QCGNN_IX(
            num_ir_qubits=num_ir_qubits,
            num_nr_qubits=num_nr_qubits,
            num_layers=num_layers,
            num_reupload=num_reupload,
            num_rotation=num_rotation,
            vqc_ansatz=vqc_ansatz,
            **kwargs
        )

        # Output is in shape (batch, NR).
        self.mlp = nn.Sequential(
            nn.Linear(num_nr_qubits, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, score_dim),
        )
    
    def forward(self, x: torch.Tensor):
        """
            x : torch.Tensor
                Shape = (N, P, 3).
        """

        with torch.no_grad():
            mask = torch.isnan(x[..., 0]) # (N, P)
            x = x.masked_fill(torch.isnan(x), 0.) # (N, P, 3)

        x = self.embed(x) # (N, P, num_rotation * n_Q)
        x = torch.atan(x)
        x = x.masked_fill(mask.unsqueeze(-1), 0.)

        x = self.phi(x, mask) # (N, n_Q)
        x = self.mlp(x)

        return x