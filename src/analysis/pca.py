import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path


def pca_analysis(X, output_dir, n_components=10):
    logging.info(f"Performing PCA analysis (n_components={n_components})...")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_, 
                alpha=0.7, edgecolor='black', label='Explained Variance')
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, n_components + 1), cumsum, 'bo-', linewidth=2, markersize=8,
                 label='Cumulative Variance')
    axes[1].axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% Variance')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'pca_analysis.png'}")
    
    pca_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(n_components)],
        'Explained_Variance_Ratio': pca.explained_variance_ratio_,
        'Cumulative_Variance': cumsum
    })
    pca_df.to_csv(output_path / 'pca_results.csv', index=False)
    logging.info(f"Saved: {output_path / 'pca_results.csv'}")
    
    return pca, X_pca