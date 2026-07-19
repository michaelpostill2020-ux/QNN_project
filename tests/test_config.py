import os
from qnn_regression.config import load_config

def test_load_config_fallback():
    # Verify we load defaults if config.yaml is absent
    config = load_config("non_existent_path.yaml")
    assert "features" in config
    assert len(config["features"]) == 3
    assert config["model"]["activation"] == "relu"