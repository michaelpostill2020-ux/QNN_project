import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
import tensorflow_quantum as tfq 
from qnn_regression.quantum import convert_to_circuit 
def plot_correlation_matrix(df, feature_cols, display_names, plot_dir):
    """Calculates and plots the Pearson correlation matrix between variables."""
    os.makedirs(plot_dir, exist_ok=True)
    
    corr_matrix = df[feature_cols].corr().values
    fig, ax = plt.subplots(figsize=(6, 5))
    
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    ticks = np.arange(0, len(feature_cols), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    

    ax.set_xticklabels(display_names, rotation=15, ha="left")
    ax.set_yticklabels(display_names)
    
    # Annotate values inside the matrix squares
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black")
            
    plt.title("Feature Correlation Matrix", pad=20, fontsize=14)
    output_path = os.path.join(plot_dir, "feature_correlation.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Correlation matrix saved to {output_path}")

def plot_loss_curves(history, fold, plot_dir, model_name="classical"):
    """Plots training vs validation loss across training epochs."""
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'{model_name.upper()} Model Loss - Fold {fold}', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_path = os.path.join(plot_dir, f"{model_name}_loss_fold_{fold}.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def plot_summary_loss_curves(all_histories, plot_dir):
    """
    Plots the training and validation loss curves across all folds 
    for classical, quantum, and hybrid models in a single master dashboard.
    """
    os.makedirs(plot_dir, exist_ok=True)
    model_types = list(all_histories.keys())
    fig, axes = plt.subplots(1, len(model_types), figsize=(6 * len(model_types), 5))
    
    if len(model_types) == 1:
        axes = [axes]
        
    for ax, m_type in zip(axes, model_types):
        histories = all_histories[m_type]
        if not histories:
            continue
            
        num_epochs = len(histories[0].history['loss'])
        epochs = range(1, num_epochs + 1)
        
        # Stack history matrices for statistical reduction
        train_losses = np.array([h.history['loss'] for h in histories])
        val_losses = np.array([h.history['val_loss'] for h in histories])
        
        # 1. Plot individual fold profiles as thin background traces
        for f_idx in range(train_losses.shape[0]):
            ax.plot(epochs, train_losses[f_idx], color='#1f77b4', alpha=0.15, linestyle='-')
            ax.plot(epochs, val_losses[f_idx], color='#ff7f0e', alpha=0.15, linestyle='--')
        
        # 2. Plot the robust mean across all data folds
        ax.plot(epochs, np.mean(train_losses, axis=0), color='#1f77b4', linewidth=2.5, label='Mean Train Loss')
        ax.plot(epochs, np.mean(val_losses, axis=0), color='#ff7f0e', linewidth=2.5, label='Mean Val Loss')
        
        ax.set_xlabel('Epochs', fontsize=11)
        ax.set_ylabel('Loss (MSE)', fontsize=11)
        ax.set_title(f"{m_type.replace('_', ' ').upper()}", fontsize=12, weight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)
        
    plt.suptitle("Master Training Convergence Profiles Across All Cross-Validation Folds", fontsize=15, weight='bold', y=1.03)
    output_path = os.path.join(plot_dir, "master_training_losses.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Master summary training loss visualization saved to {output_path}")

def run_permutation_importance(model, X_val, y_val, feature_names, fold, plot_dir, model_name="classical"):
    """
    Computes permutation feature importance by evaluating the drop in R² score 
    when individual features are randomly permuted.
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Establish baseline performance score
    if model_name == "quantum":
        import tensorflow_quantum as tfq
        from qnn_regression.quantum import convert_to_circuit
        X_val_encoded = tfq.convert_to_tensor([convert_to_circuit(row) for row in X_val])
        base_pred = model.predict(X_val_encoded).flatten()
    else:
        base_pred = model.predict(X_val).flatten()
        
    base_r2 = r2_score(y_val, base_pred)
    importances = {}
    
    # 2. Permute each column individually and record the performance drop
    for idx, name in enumerate(feature_names):
        X_permuted = X_val.copy()
        np.random.shuffle(X_permuted[:, idx])  # Break the relationship between feature and target
        
        if model_name == "quantum":
            X_perm_encoded = tfq.convert_to_tensor([convert_to_circuit(row) for row in X_permuted])
            perm_pred = model.predict(X_perm_encoded).flatten()
        else:
            perm_pred = model.predict(X_permuted).flatten()
            
        perm_r2 = r2_score(y_val, perm_pred)
        importances[name] = base_r2 - perm_r2  # Larger drop = higher importance
        
    # 3. Generate the horizontal importance bar chart
    plt.figure(figsize=(8, 4))
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=False)
    names, values = zip(*sorted_features)
    
    colors = 'cornflowerblue' if model_name == 'classical' else 'mediumseagreen'
    plt.barh(names, values, color=colors, edgecolor='black', height=0.5)
    plt.xlabel('Importance (Drop in R² Score)', fontsize=12)
    plt.title(f'{model_name.upper()} Permutation Feature Importance - Fold {fold}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')
    
    output_path = os.path.join(plot_dir, f"{model_name}_importance_fold_{fold}.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Permutation Importance plot saved to {output_path}")


def run_mae_permutation_importance(model, X_val_raw, y_val, display_names, fold, plot_dir, model_type, is_quantum=False):
    """
    Computes custom feature importance based on normalized MAE impact:
    (MAE_without_var - MAE_with_var) / MAE_without_var
    """
    print(f"Calculating custom MAE feature importance for {model_type}...")
    
    if is_quantum:
        X_val_tensor = tfq.convert_to_tensor([convert_to_circuit(row) for row in X_val_raw])
        baseline_preds = model.predict(X_val_tensor).flatten()
    else:
        baseline_preds = model.predict(X_val_raw).flatten()
        
    baseline_mae = mean_absolute_error(y_val, baseline_preds)
    importances = []
    
    for i in range(len(display_names)):
        X_permuted = X_val_raw.copy()
        np.random.seed(42)
        np.random.shuffle(X_permuted[:, i])
        
        if is_quantum:
            X_permuted_tensor = tfq.convert_to_tensor([convert_to_circuit(row) for row in X_permuted])
            permuted_preds = model.predict(X_permuted_tensor).flatten()
        else:
            permuted_preds = model.predict(X_permuted).flatten()
            
        permuted_mae = mean_absolute_error(y_val, permuted_preds)
        importance_score = (permuted_mae - baseline_mae) / permuted_mae
        importances.append(importance_score)
        
    plt.figure(figsize=(9, 5))
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), [importances[idx] for idx in indices], align='center', color='#2b5c8f')
    plt.yticks(range(len(indices)), [display_names[idx] for idx in indices])
    plt.xlabel('Normalized MAE Impact Degradation Score\n(MAE_shuffled - MAE_baseline) / MAE_shuffled')
    plt.title(f'MAE Permutation Importance - {model_type.upper()} (Fold {fold})')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"mae_importance_{model_type}_fold_{fold}.png")
    plt.savefig(plot_path)
    plt.close()