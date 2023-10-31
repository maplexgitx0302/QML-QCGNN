import m_nn

class QuantumAngle2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements, **kwargs):
        phi = m_nn.QuantumMLP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements)
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)

class QuantumElementwiseAngle2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements, **kwargs):
        phi = nn.Sequential(
            m_nn.ElementwiseLinear(in_channel=gnn_qubits),
            m_nn.QuantumMLP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements),
            )
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)

class QuantumIQP2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements, **kwargs):
        phi = m_nn.QuantumSphericalIQP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements)
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)

class QuantumElementwiseIQP2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements, **kwargs):
        phi = nn.Sequential(
            m_nn.ElementwiseLinear(in_channel=gnn_qubits),
            m_nn.QuantumSphericalIQP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements),
            )
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)