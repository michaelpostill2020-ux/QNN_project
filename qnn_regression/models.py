import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from keras.layers import Dense, Input, BatchNormalization
from keras.models import Sequential

class CustomPQC(tfq.layers.PQC):
    def __init__(self, circuit, operators, **kwargs):
        # Enforce Adjoint differentiation for maximum speedup
        adjoint_diff = tfq.differentiators.Adjoint()
        super(CustomPQC, self).__init__(circuit, operators, differentiator=adjoint_diff, **kwargs)
        self.circuit = circuit
        self.operators = operators

    def get_config(self):
        config = super().get_config()
        config.update({
            "circuit": cirq.to_json(self.circuit),
            "operators": cirq.to_json(self.operators)
        })
        return config

    @classmethod
    def from_config(cls, config):
        circuit = cirq.read_json(json_text=config["circuit"])
        operators = cirq.read_json(json_text=config["operators"])
        return cls(circuit=circuit, operators=operators)


def get_pqc_elements():
    """
    Initializes a 6-qubit, 3-layer deep Parameterized Quantum Circuit (PQC).

    """
    qubits_pqc = cirq.GridQubit.rect(1, 6)
    pqc = cirq.Circuit()
    
    # 3 layers * 6 qubits * 2 parameters per qubit (Rx, Rz) = 36 total symbols
    symbols = sympy.symbols('theta_0:36')
    sym_idx = 0
    
    # Build 3 deep variational layers
    for layer in range(3):
        # 1. Parameterized Rotations
        for i in range(6):
            pqc.append(cirq.rx(symbols[sym_idx]).on(qubits_pqc[i]))
            pqc.append(cirq.rz(symbols[sym_idx + 1]).on(qubits_pqc[i]))
            sym_idx += 2
            
        # 2. Entangling Ring Map per layer
        for i in range(5):
            pqc.append(cirq.CNOT(qubits_pqc[i], qubits_pqc[i+1]))
        pqc.append(cirq.CNOT(qubits_pqc[5], qubits_pqc[0])) # Close the ring loop
        
    # Measure expectation values across the 6 active qubits
    operators = [cirq.Z(qubits_pqc[i]) for i in range(6)]
    return pqc, operators


def build_qnn_model(pqc, operators):
    """Assembles the optimized 6-qubit QNN regression network."""
    model = Sequential([
        Input(shape=(), dtype=tf.string), 
        CustomPQC(pqc, operators), 
        BatchNormalization(), 
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def build_classical_model(input_dim=3, activation="relu"):
    """Assembles a classical deep neural network with configurable layer activations."""
    model = Sequential([
        Input(shape=(input_dim,)), 
        Dense(16, activation=activation), 
        Dense(12, activation=activation), 
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model