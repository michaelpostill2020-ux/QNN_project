import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Loads YAML configuration, falling back to safe smart-grid defaults 
    if the config file is missing.
    """
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found. Using default smart-grid configurations.")
        return {
            "data": {
                "root_file_path": "./data/smart_grid_telemetry.root",
                "output_h5_dir": "./h5",
                "model_dir": "./models",
                "plot_dir": "./plots",
                "chunk_size": 100000
            },
            "features": [
                {"name": "turbine_wind_speed_ms", "log_transform": False},
                {"name": "turbine_efficiency", "log_transform": False},
                {"name": "turbine_theoretical_mw", "log_transform": False}
            ],
            "target": {
                "name": "grid_delivered_power_mw",
                "log_transform": False
            },
            "model": {
                "activation": "relu",
                "epochs": 10,
                "batch_size": 64,
                "learning_rate": 0.001,
                "k_folds": 5
            }
        }
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)