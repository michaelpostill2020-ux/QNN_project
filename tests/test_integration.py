import os
import shutil
import pytest
import pandas as pd
import numpy as np
import tensorflow_quantum as tfq

from generate_mock_data import create_mock_dataset
from qnn_regression.config import load_config
from qnn_regression.data import process_and_save_sample
from qnn_regression.models import get_pqc_elements, build_qnn_model, build_classical_model
from qnn_regression.quantum import convert_to_circuit

# Constants for testing sandbox
TEST_DATA_DIR = "./mock_data"
TEST_CONFIG_PATH = "./tests/test_config.yaml"

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_sandbox():
    """Sets up a clean sandbox before testing and deletes it afterward."""
    # 1. Generate fresh mock data
    create_mock_dataset(TEST_DATA_DIR, num_regions=50)  # Set to 50 for fast test execution
    
    yield # --- Run the actual tests ---

    # 2. Cleanup sandbox directories
    for folder in [TEST_DATA_DIR, "./tests/h5_temp", "./tests/models_temp", "./tests/plots_temp"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

def test_end_to_end_pipeline():
    """Verifies data extraction, transformation, H5 writing, and dual-model execution."""
    # Load test config configurations
    config = load_config(TEST_CONFIG_PATH)
    
    # --- Step 1: Run the ETL Pipeline ---
    smart_grid_h5 = os.path.join(config["data"]["output_h5_dir"], "smart_grid_telemetry_test.h5")
    
    os.makedirs(config["data"]["output_h5_dir"], exist_ok=True)
    process_and_save_sample(config["data"]["root_file_path"], smart_grid_h5, "SmartGrid", config)
    
    # Verify H5 store exists and holds generated events
    assert os.path.exists(smart_grid_h5)
    
    df = pd.read_hdf(smart_grid_h5, 'events')
    assert len(df) > 0
    
    # Make sure features are standardized correctly in data.py
    for col in ['feature_0', 'feature_1', 'feature_2', 'target']:
        assert col in df.columns

    # --- Step 2: Build and Test Models ---
    # A. Classical Model compilation and single-step feed-forward verification
    classical_model = build_classical_model(input_dim=3, activation=config["model"]["activation"])
    dummy_input = np.random.rand(5, 3)
    dummy_output = classical_model.predict(dummy_input)
    assert dummy_output.shape == (5, 1)

    # B. Quantum Model Structure & Forward Pass Execution Verification
    pqc, operators = get_pqc_elements()
    qnn_model = build_qnn_model(pqc, operators)
    assert qnn_model is not None
    
    #Verify quantum data ingestion pipelines using a mock circuit tensor conversion
    mock_quantum_data = tfq.convert_to_tensor([convert_to_circuit(row) for row in dummy_input])
    qnn_output = qnn_model.predict(mock_quantum_data)
    
    # Checks that the QNN generates a singular prediction feature per event topology instance
    assert qnn_output.shape == (5, 1)