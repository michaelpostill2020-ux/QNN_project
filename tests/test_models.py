import tensorflow as tf
import tensorflow_quantum as tfq
from qnn_regression.models import get_pqc_elements, build_qnn_model, build_classical_model

def test_get_pqc_elements():
    pqc, operators = get_pqc_elements()
    assert pqc is not None
    
    # Verify operators is a list containing exactly 6 qubit measurements
    assert isinstance(operators, list)
    assert len(operators) == 6

def test_classical_model_structure():
    model = build_classical_model(input_dim=3)
    
    # Sequential groups 3 Dense layers (excluding input layer def in Keras)
    assert len(model.layers) == 3
    assert model.input_shape == (None, 3)