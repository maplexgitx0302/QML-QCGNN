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

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# The logging function.
def _log(message: str) -> None:
    """Printing function for log."""
    print(f"# ModelLog: {message}")


class Encoding:
    def parse_input(self, x: torch.Tensor):
        """Parse the input particle data x"""

        if len(x.shape) > 1:
            # The shape of `ptc_input` is (batch, 3)
            theta, phi, omega = x[:,0], x[:,1], x[:,2]
        else:
            # The shape of `ptc_input` is (3,)
            theta, phi, omega = x[0], x[1], x[2]

        return theta, phi, omega
    
    def ansatz(self):
        """Quantum circuit ansatz to be defined below."""
        pass

class PennylaneEncoding(Encoding):
    def __init__(self, num_ir_qubits: int, num_nr_qubits: int):
        self.num_ir_qubits = num_ir_qubits
        self.num_nr_qubits = num_nr_qubits
        self.num_qubits = num_ir_qubits + num_nr_qubits

    def ansatz(self, particle_input: torch.Tensor, control_values: list[int]):
        # Parse input data.
        theta, phi, omega = self.parse_input(particle_input)

        # Encode data on NR qubits.
        nr_wires = range(self.num_qubits - self.num_nr_qubits, self.num_qubits)
        for wires in nr_wires:
            # Add a Hadamard gate before the rotation gate.
            ctrl_H = qml.ctrl(qml.Hadamard, control=range(self.num_ir_qubits), control_values=control_values)
            ctrl_H(wires=wires)
            # Appl general rotation gate.
            ctrl_R = qml.ctrl(qml.Rot, control=range(self.num_ir_qubits), control_values=control_values)
            ctrl_R(theta=theta, phi=phi, omega=omega, wires=wires)


class QiskitEncoding(Encoding):
    def __init__(self, num_ir_qubits: int, num_nr_qubits: int):
        self.num_ir_qubits = num_ir_qubits
        self.num_wk_qubits = num_ir_qubits - 1
        self.num_nr_qubits = num_nr_qubits
        self.num_qubits = num_ir_qubits + self.num_wk_qubits + num_nr_qubits

    def ansatz(self, particle_input: torch.Tensor, control_values: list[int]):
        # Parse input data.
        theta, phi, omega = self.parse_input(particle_input)

        # Target wires to be encoded.
        nr_wires = range(self.num_qubits - self.num_nr_qubits, self.num_qubits)

        # See "Quantum Computation and Quantum Information" section 4.3.
        for wires in nr_wires:
            self.control_condition_transform(control_values)
            self.toffoli_tranformation()
            
            # The last working qubit becomes the control qubit.
            wk_qubit_c = self.num_ir_qubits + self.num_wk_qubits - 1
            # ctrl_H: decomposed by H = i Rx(pi) Ry(pi/2) up to a global phase.
            qml.CRY(np.pi / 2, wires=(wk_qubit_c, wires))
            qml.CRX(np.pi, wires=(wk_qubit_c, wires))
            # ctrl_R: Rot(phi, theta, omega) = Rz(omega) Ry(theta) Rz(phi).
            qml.CRZ(phi, wires=(wk_qubit_c, wires))
            qml.CRY(theta, wires=(wk_qubit_c, wires))
            qml.CRZ(omega, wires=(wk_qubit_c, wires))
            
            self.toffoli_tranformation(inverse=True)
            self.control_condition_transform(control_values)

    def control_condition_transform(self, control_values: list[int]):
        """Turn ctrl-0 to ctrl-1
        
        If control values == 0, use X-gate for transforming to 1.
        """

        for i, bit in enumerate(control_values):
            # `i` corresponds to the i-th qubit in IR.
            if bit == 0:
                qml.PauliX(wires=i)

    def toffoli_tranformation(self, inverse: bool = False):
        """Decomposition of multi-contolled gates
        
        Use Toffoli transformation for decomposition of multi-controlled gates.

        Args:
            inverse : bool
                Whether to apply inverse transformation or not.
        """

        if (not inverse) and (self.num_ir_qubits > 1):
            wk_qubit_t = self.num_ir_qubits # target qubit, also first working qubit
            qml.Toffoli(wires=(0, 1, wk_qubit_t))
        
        if inverse:
            toffoli_range = reversed(range(self.num_wk_qubits - 1))
        else:
            toffoli_range = range(self.num_wk_qubits - 1)
        
        for i in toffoli_range:
            ir_qubit_c = 2 + i # control qubit
            wk_qubit_c = self.num_ir_qubits + i # control qubit
            wk_qubit_t = self.num_ir_qubits + i + 1 # target qubit
            qml.Toffoli(wires=(ir_qubit_c, wk_qubit_c, wk_qubit_t))

        if inverse and (self.num_ir_qubits > 1):
            wk_qubit_t = self.num_ir_qubits # target qubit, also first working qubit
            qml.Toffoli(wires=(0, 1, wk_qubit_t))    


class QCGNN_IX(nn.Module):
    def __init__(
            self,
            num_ir_qubits: int,
            num_nr_qubits: int,
            num_layers: int,
            num_reupload: int,
            qdevice: str = 'default.qubit',
            qbackend: str = '',
            diff_method: str = 'best',
            shots: int = 1024,
            aggregation: str = 'add',
            noise_prob: float = 0,
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
                Number of layers in a single strongly entangling layer.
                Equivalent to the depth of a "single" VQC ansatz.
            num_reupload : int
                Number of times of the whole VQC ansatz (and data 
                reupload), at least 1.
            qdevice : str (default 'default.qubit')
                Quantum device provided by PennyLane qml.
            qbackend : str (default None)
                If using IBM quantum systems, this argument corresponds
                to the backend of the real device (usually the qdevice
                will also be specified as 'qiskit.ibmq').
            diff_method : str (default 'best')
                The method for calculating gradients. Note in real 
                devices, usually only "parameter-shift" is allowed.
            shots : int (default 1024)
                Number of measurement shots. For PennyLane ideal 
                simulators, `shots` can be ignored since the returned
                values are ideal expectation values. For IBM quantum
                systems, `shots` is default as 1024. Note large shots
                might cause crashed when using IBM quantum systems.
            aggregation : str (default 'add')
                Aggregation function ('add' or 'mean').
            noise_prob : float (default 0)
                The probability of noise.
        """

        super().__init__()

        self.num_layers = num_layers
        self.num_reupload = num_reupload
        self.aggregation = aggregation
        self.noise_prob = noise_prob
        self.diff_method = diff_method

        # Determine the encoding method.
        if ('qiskit' in qdevice) or ('qiskit' in qbackend):
            self.encoding = QiskitEncoding(num_ir_qubits, num_nr_qubits).ansatz
        else:
            self.encoding = PennylaneEncoding(num_ir_qubits, num_nr_qubits).ansatz
        
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
        _log(f"Quantum device  = {qdevice}")
        _log(f"Quantum backend = {qbackend}")
        _log(f"Qubits (IR, WK, NR) = ({num_ir_qubits, num_wk_qubits, num_nr_qubits})")

        # `wires` range that will be used later in quantum "gates".
        self.ir_wires = range(num_ir_qubits)
        self.nr_wires = range(num_ir_qubits + num_wk_qubits, num_qubits)

        # Create quantum device.
        if ('qiskit' in qdevice) and (qbackend is not None):
            # If real device -> specify backend and shots.
            self.qml_device = qml.device(qdevice, wires=num_qubits, backend=qbackend, shots=shots)
        elif qdevice == 'default.mixed':
            # Used for noise.
            self.qml_device = qml.device(qdevice, wires=num_qubits, shots=shots)
        else:
            # Other quantum simulators (including default.qubit).
            self.qml_device = qml.device(qdevice, wires=num_qubits)

        # Turn PennyLane quantum circuit into PyTorch layers.
        circuit = self.build_full_circuit()
        self.circuit = circuit
        weight_shapes = {'weights': (num_reupload, num_layers, num_nr_qubits, 3)}
        torch_layer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
        self.net = nn.Sequential(torch_layer)

    def build_full_circuit(self):
        """Build up the quantum circuit."""

        @qml.qnode(self.qml_device, diff_method=self.diff_method)
        def full_circuit(inputs, weights):
            # Build up quantum gates.
            self.circuit_evolve(inputs, weights)

            # Get observable list.
            pauli_words = self.pauli_words_of_IX_combinations()

            return [qml.expval(paruli_word) for paruli_word in pauli_words]

        return full_circuit

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

        for wires in range(self.num_qubits):
            rnd_noise = np.random.randint(2)

            if rnd_noise == 0:
                qml.DepolarizingChannel(
                    p=self.noise_prob,
                    wires=wires
                )
            elif rnd_noise == 1:
                qml.GeneralizedAmplitudeDamping(
                    gamma=self.noise_prob,
                    p=0.5,
                    wires=wires
                )

    def circuit_evolve(self, inputs, weights):
        """Quantum circuit for QCGNN."""

        # The `inputs` will be automatically reshape as (batch_size, D),
        # where D is the dimension of a single flattened data. We reshape
        # the `inputs` back to correct shape, by assuming the data
        # constructed with only 3 features (pt, eta, phi). Note that in
        # `pennylane==0.31.0` above, if original inputs shape is (N, M, D),
        # it will automatically reshape to (N*M, D).
        inputs = inputs.unflatten(dim=-1, sizes=(-1, 3))

        # `pt`, `eta`, `phi` will be padded with zero to size 2**num_ir_qubits.
        # We check how many particles in each events by (pt > 0).
        pt = inputs[..., 0]

        if pt.all() == True:
            # Typically only happens with one event.
            _log(f"**Warning** -> Using Hadamard initializaion.")
            qml.broadcast(qml.Hadamard, pattern='single', wires=self.ir_wires)
        else:
            state_vector = (pt > 0).float()
            state_vector = state_vector / torch.norm(state_vector, dim=-1, keepdim=True)
            qml.QubitStateVector(state_vector, wires=self.ir_wires)

        # Noise channel 1.
        if self.noise_prob > 0:
            self.random_noise()

        # Data-reuploading (need `num_reupload` >= 1).
        for re_idx in range(self.num_reupload):
            # Encoding data with multi-controlled gates.
            for ir_idx in range(2**self.num_ir_qubits):
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
                    self.encoding(inputs[:, ir_idx], control_values=control_values)
                else:
                    # `inputs` shape == (num_ptcs, 3), i.e., single data
                    # Note we feed in only one particle information.
                    self.encoding(inputs[ir_idx], control_values=control_values)
                # Noise channel 2.
                if self.noise_prob > 0:
                    self.random_noise()
            # Using strongly entangling layers for VQC ansatz.
            qml.StronglyEntanglingLayers(weights=weights[re_idx], wires=self.nr_wires)
            # Noise channel 3.
            if self.noise_prob > 0:
                self.random_noise()

    def forward(self, x):
        # Count number of particles (since we pad 0).
        pt = x[..., 0]
        num_ptcs = torch.sum((pt > 0).float(), axis=-1, keepdim=True)

        # Original shape of `x` is (batch, num_ptcs, 3), with 3 representing
        # features "pt", "eta" and "phi". Since PennyLane confuses with the
        # dimension of batch and features, we need to reshape `x` as
        # (batch, num_ptcs * 3), or (num_ptcs * 3) for single data only.
        x = torch.flatten(x, start_dim=-2, end_dim=-1)

        # Pass `x` through the quantum circuits, the output shape will be
        # (batch, (2**IR) * NR), where IR/NR = num_(ir/nr)_qubits respectively.
        x = self.net(x)

        # Reshape the measurement outputs to (batch, (2**IR), NR).
        if self.__class__.__name__ == 'QCGNN_IX':
            sizes = (2 ** self.num_ir_qubits, self.num_nr_qubits)
            x = torch.unflatten(x, dim=-1, sizes=sizes)
            # Transpose to shape (batch, NR, (2**IR)).
            x = x.mT
            # Summing up (2**IR) {I,X} combinations in IR.
            x = torch.sum(x, dim=-1)

        if self.aggregation == 'add':
            x = x * num_ptcs
        elif self.aggregation == 'mean':
            # Remember in classical, we have 2 MEAN aggregations, one for node,
            # and one for graph, which means we have total (1/N)^2 factor. So
            # we divide with an additional factor 1/N here.
            x = x / num_ptcs

        # `x` is now in shape (batch, NR).
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
        def full_circuit(inputs, weights):
            """
            Returns:
                A tensor constructed by concatenating:
                - IR_prob: Probalities of IR state.
                - NR_expval: Expectation values of NR.
            """
            # Build up quantum gates.
            self.circuit_evolve(inputs, weights)
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
            score_dim: int,
            **kwargs
    ):
        """The quantum model that will be mainly used.

        This quantum model is based on the `QCGNN_IX`.

        Args:
            score_dim : int
                Dimension of the final score output.
        """
        
        super().__init__()

        self.phi = QCGNN_IX(
            num_ir_qubits=num_ir_qubits,
            num_nr_qubits=num_nr_qubits,
            num_layers=num_layers,
            num_reupload=num_reupload,
            **kwargs
        )

        # Output is in shape (batch, NR).
        self.mlp = nn.Linear(num_nr_qubits, score_dim)
    
    def forward(self, x):
        x = self.phi(x)
        x = self.mlp(x)
        return x