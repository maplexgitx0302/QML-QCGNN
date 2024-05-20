"""Quantum submodels."""

import pennylane as qml
import torch.nn as nn

# Solved in newer version.
# See https://discuss.pennylane.ai/t/qml-prod-vs-direct-operators-product/3873
# qml.operation.enable_new_opmath()

# The logging function.
def _log(message: str) -> None:
    """Printing function for log."""
    print(f"# ModelLog: {message}")


class QuantumMLP(nn.Module):
    def __init__(
            self,
            num_qubits: int,
            num_layers: int,
            num_reupload: int,
            measurements: list[int, str],
            qdevice: str = 'default.qubit'
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
                Number of times of the whole VQC ansatz (and data 
                reupload), at least 1.
            measurements : list[int, str]
                A list containing tuples with two elements. The first
                element is an integer, which corresponds to the index of
                the qubit to be measured. The second element corresponds
                to the measurement basis, with value as a string "I", 
                "X", "Y" or "Z".
            qdevice : str ("default.qubit")
                Quantum device provided by PennyLane qml.
        """

        super().__init__()

        pauli_matrices = {
            'I': qml.Identity,
            'X': qml.PauliX,
            'Y': qml.PauliY,
            'Z': qml.PauliY,
        }
        
        # Quantum circuit.
        @qml.qnode(qml.device(qdevice, wires=num_qubits))
        def circuit(inputs, weights):

            # Data reupload.
            for i in range(num_reupload):

                # Data embedding.
                qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation='Y')

                # VQC layer with parameters.
                qml.StronglyEntanglingLayers(weights=weights[i], wires=range(num_qubits))
                
            # Turn measurements into observable list.
            observable_list = []
            for wires, pauli_str in measurements:
                pauli_observable = pauli_matrices[pauli_str]
                observable = pauli_observable(wires=wires)
                observable_list.append(observable)
            
            return [qml.expval(observable) for observable in observable_list]

        # Turn the quantum circuit into a torch layer.
        weight_shapes = {'weights': (num_reupload, num_layers, num_qubits, 3)}
        self.net = nn.Sequential(qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes))

    def forward(self, x):
        return self.net(x)