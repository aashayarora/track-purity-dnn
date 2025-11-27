import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import expit
from pathlib import Path

def robustness_testing(model, X, y_true, output_dir, 
                      noise_levels=[0.01, 0.05, 0.1], n_samples=5000, batch_size=32768):
    logging.info("\nPerforming robustness testing...")
    
    n_samples = min(n_samples, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y_true[indices]
    
    results = []
    
    y_pred_baseline_logits = model.predict(X_subset, batch_size=batch_size)
    y_pred_baseline_proba = expit(y_pred_baseline_logits)
    y_pred_baseline_binary = (y_pred_baseline_proba >= 0.5).astype(int)
    
    acc_baseline = accuracy_score(y_subset, y_pred_baseline_binary)
    auc_baseline = roc_auc_score(y_subset, y_pred_baseline_proba)
    
    results.append({
        'Noise Level': 0.0, 
        'Accuracy': acc_baseline,
        'ROC-AUC': auc_baseline,
        'Accuracy Decrease': 0.0,
        'AUC Decrease': 0.0
    })

    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, X_subset.shape)
        X_noisy = X_subset + noise
        
        y_pred_noisy_logits = model.predict(X_noisy, batch_size=batch_size)
        y_pred_noisy_proba = expit(y_pred_noisy_logits)
        y_pred_noisy_binary = (y_pred_noisy_proba >= 0.5).astype(int)
        
        acc_noisy = accuracy_score(y_subset, y_pred_noisy_binary)
        auc_noisy = roc_auc_score(y_subset, y_pred_noisy_proba)
        
        acc_decrease = ((acc_baseline - acc_noisy) / acc_baseline) * 100
        auc_decrease = ((auc_baseline - auc_noisy) / auc_baseline) * 100
        
        results.append({
            'Noise Level': noise_level,
            'Accuracy': acc_noisy,
            'ROC-AUC': auc_noisy,
            'Accuracy Decrease': acc_decrease,
            'AUC Decrease': auc_decrease
        })
        logging.info(f"  Noise {noise_level:.2f}: Accuracy={acc_noisy:.4f} ({acc_decrease:+.2f}%), "
                  f"ROC-AUC={auc_noisy:.4f} ({auc_decrease:+.2f}%)")
    
    results_df = pd.DataFrame(results)
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(results_df['Noise Level'], results_df['Accuracy'], 
                    'bo-', linewidth=2, markersize=8, label='Accuracy')
    axes[0, 0].set_xlabel('Noise Level (Std)', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Accuracy vs Input Noise', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    axes[0, 1].plot(results_df['Noise Level'], results_df['ROC-AUC'], 
                    'go-', linewidth=2, markersize=8, label='ROC-AUC')
    axes[0, 1].set_xlabel('Noise Level (Std)', fontsize=12)
    axes[0, 1].set_ylabel('ROC-AUC', fontsize=12)
    axes[0, 1].set_title('ROC-AUC vs Input Noise', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].plot(results_df['Noise Level'][1:], results_df['Accuracy Decrease'][1:], 
                    'ro-', linewidth=2, markersize=8, label='Accuracy Decrease')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    axes[1, 0].set_xlabel('Noise Level (Std)', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy Decrease (%)', fontsize=12)
    axes[1, 0].set_title('Accuracy Degradation', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(results_df['Noise Level'][1:], results_df['AUC Decrease'][1:], 
                    'mo-', linewidth=2, markersize=8, label='ROC-AUC Decrease')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    axes[1, 1].set_xlabel('Noise Level (Std)', fontsize=12)
    axes[1, 1].set_ylabel('ROC-AUC Decrease (%)', fontsize=12)
    axes[1, 1].set_title('ROC-AUC Degradation', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'robustness_testing.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'robustness_testing.png'}")
    
    results_df.to_csv(output_path / 'robustness_results.csv', index=False)
    logging.info(f"Saved: {output_path / 'robustness_results.csv'}")
    
    return results_df