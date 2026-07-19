import pytest
import cirq
import numpy as np
from qnn_regression.quantum import convert_to_circuit

def test_convert_to_circuit_valid():
    # Provide exactly 3 classical feature values as input
    mock_features = [0.5, -0.2, 1.2]
    circuit = convert_to_circuit(mock_features)
    
    # Assert return type remains a valid Cirq Circuit object
    assert isinstance(circuit, cirq.Circuit)
    
    # Extract and sort qubits to verify the= topology
    all_qubits = sorted(list(circuit.all_qubits()))
    
    expected_qubits = 6 
    assert len(all_qubits) == expected_qubits
    
    # Dynamic topology checking to handle both LineQubit and GridQubit redesigns
    first_qubit = all_qubits[0]
    last_qubit = all_qubits[-1]
    
    if isinstance(first_qubit, cirq.LineQubit):
        assert first_qubit == cirq.LineQubit(0)
        assert last_qubit == cirq.LineQubit(expected_qubits - 1)
    elif isinstance(first_qubit, cirq.GridQubit):
        assert first_qubit == cirq.GridQubit(0, 0)
        assert last_qubit == cirq.GridQubit(0, expected_qubits - 1)
    else:
        pytest.fail(f"Unexpected qubit layout type detected: {type(first_qubit)}")

def test_convert_to_circuit_invalid_length():
    # Verify strict safety guardrails against broken classical data shapes
    with pytest.raises(ValueError, match="must have exactly 3 features"):
        convert_to_circuit([1.0, 2.0])
        
    with pytest.raises(ValueError, match="must have exactly 3 features"):
        convert_to_circuit([1.0, 2.0, 3.0, 4.0])