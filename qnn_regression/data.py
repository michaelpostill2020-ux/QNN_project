import os
import glob
import uproot
import numpy as np
import pandas as pd
import awkward as ak

def process_and_save_sample(file_wildcard_path, h5_path, label, config):
    """
    Processes ROOT files dynamically based on your YAML configuration.
    Outputs HDF5 sets with standard names to isolate learning pipeline from branch typos.
    """
    print(f"Starting conversion for {label} data from: {file_wildcard_path}")
    
    features_cfg = config.get("features", [])
    target_cfg = config.get("target", {})
    chunk_size = config.get("data", {}).get("chunk_size", 100000)
    
    if len(features_cfg) != 3:
        raise ValueError("Config mismatch: You must specify exactly 3 features.")
        
    # Build array of variables by query from the tree
    variables_to_read = [feat["name"] for feat in features_cfg] + [target_cfg["name"]]
    
    try:
        with pd.HDFStore(h5_path, mode='w', complevel=9, complib='blosc') as store:
            print(f"Initialized {h5_path}")
            
        total_clusters_saved = 0
        total_events_processed = 0
        
        file_list = glob.glob(file_wildcard_path)
        if not file_list:
            raise FileNotFoundError(f"No files found matching path: {file_wildcard_path}")
            
        for file_path in file_list:
            with uproot.open(file_path) as file:
                if "EventTree" not in file:
                    continue
                tree = file["EventTree"]
                if tree.num_entries == 0:
                    continue
                    
                print(f"Processing: {os.path.basename(file_path)}")
                for chunk in tree.iterate(variables_to_read, library="ak", step_size=chunk_size):
                    total_events_processed += len(chunk)
                    
                    # Flatten arrays and build dynamic masks
                    raw_arrays = {}
                    for feat in features_cfg:
                        raw_arrays[feat["name"]] = ak.to_numpy(ak.flatten(chunk[feat["name"]]))
                    raw_arrays[target_cfg["name"]] = ak.to_numpy(ak.flatten(chunk[target_cfg["name"]]))
                    
                    # Create a composite filter mask
                    valid_mask = np.ones_like(raw_arrays[target_cfg["name"]], dtype=bool)
                    for feat in features_cfg:
                        if feat.get("log_transform", False):
                            valid_mask &= (raw_arrays[feat["name"]] > 0)
                    if target_cfg.get("log_transform", False):
                        valid_mask &= (raw_arrays[target_cfg["name"]] > 0)
                        
                    # Apply transformations & standardise column mappings
                    df_payload = {}
                    for idx, feat in enumerate(features_cfg):
                        feat_arr = raw_arrays[feat["name"]][valid_mask]
                        if feat.get("log_transform", False):
                            df_payload[f"feature_{idx}"] = np.log10(feat_arr)
                        else:
                            df_payload[f"feature_{idx}"] = feat_arr
                            
                    # Target mapping
                    target_arr = raw_arrays[target_cfg["name"]][valid_mask]
                    if target_cfg.get("log_transform", False):
                        df_payload["target"] = np.log10(target_arr)
                    else:
                        df_payload["target"] = target_arr
                        
                    num_clusters_saved = len(target_arr)
                    total_clusters_saved += num_clusters_saved
                    
                    if num_clusters_saved == 0:
                        continue
                        
                    df_chunk = pd.DataFrame(df_payload).astype('float64')
                    with pd.HDFStore(h5_path, mode='a', complevel=9, complib='blosc') as store:
                        store.append('events', df_chunk, format='table', data_columns=True)
                        
        print(f"Success! Saved {total_clusters_saved} clusters (Processed {total_events_processed} events).")
    except Exception as e:
        print(f"ETL pipeline execution failed: {e}")