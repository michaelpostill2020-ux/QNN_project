import cirq
import numpy as np

def convert_to_circuit(data):
    """
    Encodes 3 classical features onto 6 qubits using Rx rotation gates.
    Features are cross-entangled using a global CNOT chain to expand the phase space.
    """
    if len(data) != 3:
        raise ValueError("Input data must have exactly 3 features.")
        
    qubits = cirq.GridQubit.rect(1, 6)
    circuit = cirq.Circuit()
    
    # --- Feature Map Assignment (2 qubits per feature) ---
    feature_0 = data[0] * np.pi
    feature_1 = data[1] * np.pi
    feature_2 = data[2] * np.pi

    # Feature 0 -> Qubits 0, 1
    circuit.append(cirq.rx(feature_0).on(qubits[0]))
    circuit.append(cirq.rx(feature_0).on(qubits[1]))

    # Feature 1 -> Qubits 2, 3
    circuit.append(cirq.rx(feature_1).on(qubits[2]))
    circuit.append(cirq.rx(feature_1).on(qubits[3]))

    # Feature 2 -> Qubits 4, 5
    circuit.append(cirq.rx(feature_2).on(qubits[4]))
    circuit.append(cirq.rx(feature_2).on(qubits[5]))

    # --- Global Cross-Feature Entanglement ---
    # Connects distinct features together chronologically before the PQC
    for i in range(5):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

    return circuit