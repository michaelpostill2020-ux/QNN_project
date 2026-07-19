import os
import argparse
import numpy as np
import awkward as ak
import uproot

def generate_grid_telemetry(num_regions=1000):
    """
    Generates a single unified dataset of variable-length wind farm telemetry.
    Contains a mix of low, medium, and high wind conditions for robust regression training.
    """
    # 1. Varying number of wind turbines per grid region (3 to 15)
    turbines_per_region = np.random.randint(3, 16, size=num_regions)
    total_turbines = int(np.sum(turbines_per_region))

    # 2. Wind Speed: Weibull distribution
    # We introduce natural variability across the whole dataset
    wind_scale = np.random.uniform(8.0, 16.0, size=total_turbines)
    wind_speed = np.random.weibull(a=2.2, size=total_turbines) * wind_scale + 1.0
    
    # 3. Turbine Efficiency: Beta distribution (models mechanical wear/maintenance)
    equipment_efficiency = np.random.beta(a=6.0, b=1.2, size=total_turbines)
    
    # 4. Theoretical Kinetic Power: Implements the cubic fluid dynamics law (P ∝ v^3)
    # Air density (rho) ≈ 1.225 kg/m^3. Scaling factor applied for realistic MW output.
    theoretical_power = 0.5 * 1.225 * equipment_efficiency * (wind_speed ** 3) * 0.001
    
    # 5. Target: Actual Grid-Injected Power
    # Introduces ~9% average transmission loss/wake effects with Gaussian noise
    actual_power_delivered = theoretical_power * np.random.normal(loc=0.91, scale=0.05, size=total_turbines)
    actual_power_delivered = np.clip(actual_power_delivered, 0.0, None)

    # Convert flat arrays to jagged (region-by-region / event-by-event) structures
    ak_wind = ak.unflatten(wind_speed, turbines_per_region)
    ak_eff = ak.unflatten(equipment_efficiency, turbines_per_region)
    ak_theo = ak.unflatten(theoretical_power, turbines_per_region)
    ak_target = ak.unflatten(actual_power_delivered, turbines_per_region)

    return {
        "turbine_wind_speed_ms": ak_wind,
        "turbine_efficiency": ak_eff,
        "turbine_theoretical_mw": ak_theo,
        "grid_delivered_power_mw": ak_target
    }

def create_mock_dataset(output_dir, num_regions):
    """Creates a single comprehensive ROOT file for training and evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "smart_grid_telemetry.root")

    print(f"Generating {num_regions} regions of synthetic grid telemetry...")
    dataset = generate_grid_telemetry(num_regions=num_regions)

    print(f"Writing unified dataset to ROOT file -> {output_path}")
    with uproot.recreate(output_path) as f:
        f["EventTree"] = dataset

    print("Dataset successfully generated!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Smart-Grid ROOT Data Generator")
    parser.add_argument('--outdir', default='./mock_grid_data', help="Directory to save ROOT file")
    parser.add_argument('--regions', type=int, default=1000, help="Total number of grid regions to simulate")
    args = parser.parse_args()
    
    create_mock_dataset(args.outdir, args.regions)