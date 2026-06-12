#!/usr/bin/env python3
"""
Visualization script for GEARS score optimization results.
Creates plots comparing different lambda values and score distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from glob import glob

def plot_lambda_comparison(results_dir, output_file='lambda_comparison.png'):
    """
    Plot comparison of different lambda values.
    """
    # Find comparison summary
    comparison_files = glob(os.path.join(results_dir, 'lambda_comparison_*/comparison_summary.json'))
    
    if not comparison_files:
        print("No comparison results found. Run with --compare first.")
        return
    
    # Load latest comparison
    latest_file = sorted(comparison_files)[-1]
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    lambdas = [r['score_lambda'] for r in results]
    mses = [r['test_metrics']['mse'] for r in results]
    pearsons = [r['test_metrics']['pearson'] for r in results]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE plot
    ax1.plot(lambdas, mses, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Score Lambda (λ)', fontsize=12)
    ax1.set_ylabel('Test MSE', fontsize=12)
    ax1.set_title('Test MSE vs Score Weight', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmin(mses)
    ax1.scatter([lambdas[best_idx]], [mses[best_idx]], 
               color='red', s=200, marker='*', zorder=5,
               label=f'Best: λ={lambdas[best_idx]:.2f}')
    ax1.legend()
    
    # Pearson plot
    ax2.plot(lambdas, pearsons, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Score Lambda (λ)', fontsize=12)
    ax2.set_ylabel('Test Pearson Correlation', fontsize=12)
    ax2.set_title('Test Pearson vs Score Weight', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmax(pearsons)
    ax2.scatter([lambdas[best_idx]], [pearsons[best_idx]], 
               color='red', s=200, marker='*', zorder=5,
               label=f'Best: λ={lambdas[best_idx]:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Lambda comparison plot saved to: {output_file}")
    plt.close()


def plot_score_distribution(h5ad_path, output_file='score_distribution.png'):
    """
    Plot distribution of perturbation scores.
    """
    import scanpy as sc
    
    # Load data
    adata = sc.read_h5ad(h5ad_path)
    
    if 'perturbation_score' not in adata.obs.columns:
        print("No perturbation_score column found in h5ad file")
        return
    
    scores = adata.obs['perturbation_score'].dropna()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall distribution
    ax = axes[0, 0]
    ax.hist(scores, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Perturbation Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Score Distribution', fontsize=14, fontweight='bold')
    ax.axvline(scores.mean(), color='red', linestyle='--', 
              label=f'Mean: {scores.mean():.3f}')
    ax.legend()
    
    # Box plot by condition
    ax = axes[0, 1]
    
    # Get top perturbations by number of cells
    top_perts = adata.obs['condition'].value_counts().head(10).index
    subset = adata.obs[adata.obs['condition'].isin(top_perts)]
    
    if len(subset) > 0:
        subset.boxplot(column='perturbation_score', by='condition', ax=ax)
        ax.set_xlabel('Perturbation', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Scores by Perturbation (Top 10)', fontsize=14, fontweight='bold')
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')
    
    # KDE plot
    ax = axes[1, 0]
    from scipy import stats
    density = stats.gaussian_kde(scores)
    x_range = np.linspace(scores.min(), scores.max(), 200)
    ax.plot(x_range, density(x_range), linewidth=2, color='#A23B72')
    ax.fill_between(x_range, density(x_range), alpha=0.3, color='#A23B72')
    ax.set_xlabel('Perturbation Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Density Distribution', fontsize=14, fontweight='bold')
    
    # Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Score Statistics
    ─────────────────────────
    Total Cells:      {len(adata.obs):,}
    Scored Cells:     {len(scores):,}
    Control Cells:    {(adata.obs['perturbation_score'] == 0).sum():,}
    
    Min:              {scores.min():.4f}
    Max:              {scores.max():.4f}
    Mean:             {scores.mean():.4f}
    Median:           {scores.median():.4f}
    Std:              {scores.std():.4f}
    
    Quartiles:
      Q1 (25%):       {scores.quantile(0.25):.4f}
      Q2 (50%):       {scores.quantile(0.50):.4f}
      Q3 (75%):       {scores.quantile(0.75):.4f}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Score distribution plot saved to: {output_file}")
    plt.close()


def plot_training_history(results_dir, lambda_value, output_file='training_history.png'):
    """
    Plot training history for a specific lambda value.
    """
    metrics_file = os.path.join(results_dir, f'lambda_{lambda_value}', 'metrics.json')
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    history = metrics['training_history']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], label='Train Loss', 
           linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', 
           linewidth=2, marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training History (λ = {lambda_value})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {output_file}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize GEARS score optimization results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    parser.add_argument('--h5ad_path', type=str, 
                       default='./perturb_processed_with_scores.h5ad',
                       help='Path to h5ad file with scores')
    parser.add_argument('--lambda_value', type=float, default=0.3,
                       help='Lambda value for training history plot')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Generating visualization plots...")
    print("="*60)
    
    # Plot lambda comparison
    print("\n1. Lambda comparison plot...")
    plot_lambda_comparison(
        args.results_dir,
        os.path.join(args.output_dir, 'lambda_comparison.png')
    )
    
    # Plot score distribution
    print("\n2. Score distribution plot...")
    plot_score_distribution(
        args.h5ad_path,
        os.path.join(args.output_dir, 'score_distribution.png')
    )
    
    # Plot training history
    print("\n3. Training history plot...")
    plot_training_history(
        args.results_dir,
        args.lambda_value,
        os.path.join(args.output_dir, 'training_history.png')
    )
    
    print("\n" + "="*60)
    print(f"All plots saved to: {args.output_dir}")
    print("="*60)
