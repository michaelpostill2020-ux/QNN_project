import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from qnn_regression.utils import run_mae_permutation_importance, plot_summary_loss_curves
from qnn_regression.config import load_config
from qnn_regression.data import process_and_save_sample
from qnn_regression.models import get_pqc_elements, build_qnn_model, build_classical_model
from qnn_regression.utils import plot_correlation_matrix, plot_loss_curves
from qnn_regression.quantum import convert_to_circuit 


def main():
    parser = argparse.ArgumentParser(description="Configurable Hybrid QML Smart Grid Regression")
    parser.add_argument('--config', default='config.yaml', help="Path to config file")
    parser.add_argument('--convert', action='store_true', help="Convert ROOT data to H5")
    parser.add_argument('--evaluate', action='store_true', help="Compute combined ensemble test set MAE & compare configurations")
    parser.add_argument('--train', action='store_true', help="Train Baseline Classical, Quantum, and Hybrid models")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["data"]["output_h5_dir"]
    model_dir = config["data"]["model_dir"]
    plot_dir = config["data"]["plot_dir"]
    
    unified_h5 = os.path.join(output_dir, "smart_grid_telemetry.h5")
    feature_cols = ['feature_0', 'feature_1', 'feature_2']
    target_col = 'target'
    input_dim_base = len(feature_cols)

    # Pre-calculate feature display labels globally
    display_names_base = []
    for feat in config.get("features", []):
        name = feat["name"]
        display_names_base.append(f"log10({name})" if feat.get("log_transform", False) else name)
    if not display_names_base:
        display_names_base = feature_cols

    display_names_hybrid = display_names_base + ["QNN_Prediction_Feature"]

    if args.convert:
        os.makedirs(output_dir, exist_ok=True)
        process_and_save_sample(config["data"]["root_file_path"], unified_h5, "SmartGrid", config)

    # ==========================================================================
    # 2. Consolidated Evaluation & Comparison Phase
    # ==========================================================================
    if args.evaluate:
        print("\n==========================================================")
        print("🎯 RUNNING COMPREHENSIVE HOLDOUT ENSEMBLE EVALUATION")
        print("==========================================================\n")
        
        if not os.path.exists(unified_h5):
            raise FileNotFoundError(f"Missing processed data stack: {unified_h5}. Run with --convert first.")
            
        df = pd.read_hdf(unified_h5, 'events')
        X = df[feature_cols].values
        y = df[target_col].values
        
        _, X_test_global, _, y_test_global = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Encoding test data stream into quantum tensors...")
        X_test_quantum = tfq.convert_to_tensor([convert_to_circuit(row) for row in X_test_global])
        
        k_folds = config["model"].get("k_folds", 5)
        baseline_ensemble_preds = np.zeros(len(y_test_global))
        qnn_ensemble_preds = np.zeros(len(y_test_global))
        hybrid_ensemble_preds = np.zeros(len(y_test_global))
        successful_folds = 0
        
        for fold in range(k_folds):
            f_num = fold + 1
            baseline_path = os.path.join(model_dir, f"baseline_classical_regressor_fold_{f_num}.h5")
            qnn_weights_path = os.path.join(model_dir, f"qnn_weights_fold_{f_num}.h5")
            hybrid_model_path = os.path.join(model_dir, f"hybrid_classical_regressor_fold_{f_num}.h5")
            
            if not (os.path.exists(baseline_path) and os.path.exists(qnn_weights_path) and os.path.exists(hybrid_model_path)):
                print(f"⚠️ Missing checkpoints for Fold {f_num}, skipping from tracking.")
                continue
                
            # 1. Pure Classical Evaluation Track
            baseline_model = tf.keras.models.load_model(baseline_path)
            baseline_preds = baseline_model.predict(X_test_global, verbose=0).flatten()
            baseline_ensemble_preds += baseline_preds
            
            # 2. Pure Quantum Evaluation Track
            pqc, operators = get_pqc_elements()
            qnn_model = build_qnn_model(pqc, operators)
            qnn_model.load_weights(qnn_weights_path)
            qnn_preds = qnn_model.predict(X_test_quantum, verbose=0).flatten()
            qnn_ensemble_preds += qnn_preds
            
            # 3. Hybrid Quantum-Classical Evaluation Track
            X_test_hybrid = np.hstack([X_test_global, qnn_preds.reshape(-1, 1)])
            hybrid_model = tf.keras.models.load_model(hybrid_model_path)
            hybrid_preds = hybrid_model.predict(X_test_hybrid, verbose=0).flatten()
            hybrid_ensemble_preds += hybrid_preds
            
            successful_folds += 1
            
        if successful_folds > 0:
            mean_qnn_preds = qnn_ensemble_preds / successful_folds
            final_baseline_mae = mean_absolute_error(y_test_global, baseline_ensemble_preds / successful_folds)
            final_qnn_mae = mean_absolute_error(y_test_global, mean_qnn_preds)
            final_hybrid_mae = mean_absolute_error(y_test_global, hybrid_ensemble_preds / successful_folds)
            
            # Inject the ensembled QNN Feature back into a matrix slice for plotting
            eval_corr_df = pd.DataFrame(X_test_global, columns=feature_cols)
            eval_corr_df["QNN_Prediction_Feature"] = mean_qnn_preds
            
            extended_feature_cols = feature_cols + ["QNN_Prediction_Feature"]
            
            print("\nGenerating unified feature correlation matrix including the QNN Feature space...")
            plot_correlation_matrix(eval_corr_df, extended_feature_cols, display_names_hybrid, plot_dir)
            
            reduction = (final_baseline_mae - final_hybrid_mae) / final_baseline_mae * 100
            
            print("\n" + "="*60)
            print(f" 🏁 FINAL HOLDOUT PERFORMANCE BENCHMARK ({successful_folds}-Fold Ensembles)")
            print("="*60)
            print(f" 1. Pure Classical DNN MAE ({input_dim_base} Features):    {final_baseline_mae:.5f}")
            print(f" 2. Standalone Quantum PQC MAE:            {final_qnn_mae:.5f}")
            print(f" 3. Hybrid Quantum-Classical MAE ({input_dim_base + 1} Feat): {final_hybrid_mae:.5f}")
            print("-"*60)
            if reduction > 0:
                print(f" 📈 SUCCESS: Hybrid model reduces MAE by {reduction:.2f}% over Pure Classical.")
            else:
                print(f" 📉 NOTICE: Hybrid model changed MAE by {reduction:.2f}% vs Pure Classical.")
            print("="*60 + "\n")
        else:
            print("❌ Evaluation failed. Ensure your baseline, qnn, and hybrid checkpoint directories are populated.")

    # ==========================================================================
    # 3. Hybrid Model Training Phase
    # ==========================================================================
    if args.train:
        print("\n=============================================")
        print("🚀 STARTING HYBRID PIPELINE WITH TEST ISOLATION")
        print("=============================================\n")
        
        if not os.path.exists(unified_h5):
            raise FileNotFoundError(f"Missing processed data stack: {unified_h5}. Run with --convert first.")

        df = pd.read_hdf(unified_h5, 'events')
        X = df[feature_cols].values
        y = df[target_col].values

        X_train_val_pool, _, y_train_val_pool, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Encoding entire classical data pool into quantum circuits...")
        X_pool_quantum = tfq.convert_to_tensor([convert_to_circuit(row) for row in X_train_val_pool])

        k_folds = config["model"].get("k_folds", 5)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        histories_tracker = {
            "classical_baseline": [],
            "quantum": [],
            "classical_hybrid": []
        }
        
        print(f"Data pool scaled down to {len(X_train_val_pool)} items for cross-validation splits.")
        os.makedirs(model_dir, exist_ok=True)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val_pool)):
            print(f"\n=========================== FOLD {fold + 1}/{k_folds} ===========================")
            
            X_train, X_val = X_train_val_pool[train_idx], X_train_val_pool[val_idx]
            y_train, y_val = y_train_val_pool[train_idx], y_train_val_pool[val_idx]

            X_train_quantum = tf.gather(X_pool_quantum, train_idx)
            X_val_quantum = tf.gather(X_pool_quantum, val_idx)

            # --- PHASE 1: BASELINE PURE CLASSICAL DNN ---
            print(f"\n--- [Fold {fold + 1}] Training Baseline Pure Classical DNN ---")
            baseline_model = build_classical_model(input_dim=input_dim_base, activation=config["model"]["activation"])
            baseline_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config["model"]["learning_rate"]),
                loss='mean_squared_error', metrics=['mae']
            )
            baseline_history = baseline_model.fit(
                X_train, y_train, validation_data=(X_val, y_val),
                epochs=config["model"]["epochs"], batch_size=config["model"]["batch_size"], verbose=1
            )
            histories_tracker["classical_baseline"].append(baseline_history)
            plot_loss_curves(baseline_history, fold + 1, plot_dir, "classical_baseline")
            
            baseline_model_path = os.path.join(model_dir, f"baseline_classical_regressor_fold_{fold + 1}.h5")
            baseline_model.save(baseline_model_path)

            # --- PHASE 2: QUANTUM TRAINING & EVALUATION ---
            print(f"\n--- [Fold {fold + 1}] Training Quantum Front-End (QNN) ---")
            pqc, operators = get_pqc_elements()
            qnn_model = build_qnn_model(pqc, operators)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=config["model"]["learning_rate"])
            qnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

            qnn_history = qnn_model.fit(
                x=X_train_quantum, y=y_train, validation_data=(X_val_quantum, y_val),
                epochs=config["model"]["epochs"], batch_size=config["model"]["batch_size"], verbose=1
            )
            histories_tracker["quantum"].append(qnn_history)
            plot_loss_curves(qnn_history, fold + 1, plot_dir, "quantum")

            qnn_weights_path = os.path.join(model_dir, f"qnn_weights_fold_{fold + 1}.h5")
            qnn_model.save_weights(qnn_weights_path)

            # --- PHASE 3: INJECT QNN OUTPUT INTO DATA MATRIX ---
            print(f"\nExtracting latent quantum prediction attributes for fold matrices...")
            qnn_pred_train = qnn_model.predict(X_train_quantum).reshape(-1, 1)
            qnn_pred_val = qnn_model.predict(X_val_quantum).reshape(-1, 1)

            X_train_hybrid = np.hstack([X_train, qnn_pred_train])
            X_val_hybrid = np.hstack([X_val, qnn_pred_val])

            # --- PHASE 4: HYBRID CLASSICAL DNN TRAINING ---
            print(f"\n--- [Fold {fold + 1}] Training Hybrid Classical Deep Neural Network ---")
            classical_model = build_classical_model(input_dim=input_dim_base + 1, activation=config["model"]["activation"])
            classical_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config["model"]["learning_rate"]),
                loss='mean_squared_error', metrics=['mae']
            )
            classical_history = classical_model.fit(
                X_train_hybrid, y_train, validation_data=(X_val_hybrid, y_val),
                epochs=config["model"]["epochs"], batch_size=config["model"]["batch_size"], verbose=1
            )
            histories_tracker["classical_hybrid"].append(classical_history)
            plot_loss_curves(classical_history, fold + 1, plot_dir, "classical_hybrid")
            
            classical_model_path = os.path.join(model_dir, f"hybrid_classical_regressor_fold_{fold + 1}.h5")
            classical_model.save(classical_model_path)

        # Plot master comparative line chart summaries across all tracking datasets
        plot_summary_loss_curves(histories_tracker, plot_dir)

        print("\n=============================================")
        print("🎉 TRAINING COMPLETE. EXECUTABLE WITH --evaluate FOR STATISTICAL COMPILATION.")
        print("=============================================")

if __name__ == "__main__":
    main()