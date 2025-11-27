import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

def correlation_heatmap(X, y_true, feature_names, output_dir, top_n=20):
    correlations = []
    for i in range(X.shape[1]):
        corr, _ = pearsonr(X[:, i], y_true)
        correlations.append(abs(corr))
    
    top_indices = np.argsort(correlations)[-top_n:]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    X_top = X[:, top_indices]
    corr_matrix = np.corrcoef(X_top.T)
    
    output_path = Path(output_dir)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, xticklabels=top_feature_names, yticklabels=top_feature_names,
               cmap='coolwarm', center=0, annot=False, fmt='.2f', 
               cbar_kws={'label': 'Correlation'})
    plt.title(f'Feature Correlations', 
             fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'correlation_heatmap.png'}")
    
    target_corr_df = pd.DataFrame({
        'Feature': feature_names,
        'Correlation_with_Target': [pearsonr(X[:, i], y_true)[0] 
                                   for i in range(X.shape[1])]
    }).sort_values('Correlation_with_Target', key=abs, ascending=False)
    
    target_corr_df.to_csv(output_path / 'feature_target_correlation.csv', index=False)
    logging.info(f"Saved: {output_path / 'feature_target_correlation.csv'}")
    
    return corr_matrix, top_feature_names