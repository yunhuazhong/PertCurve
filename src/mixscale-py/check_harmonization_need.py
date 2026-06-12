"""
Quick check to determine if harmonization is needed for your data.

This script compares cell-type distributions between NT and perturbed cells.
If distributions are very different, harmonization may be beneficial.
"""

import scanpy as sc
import pandas as pd
import numpy as np

def check_harmonization_need(adata, labels='gene', nt_class_name='NT', split_by=None):
    """
    Check if harmonization is needed by comparing cell-type distributions.
    
    Parameters
    ----------
    adata : AnnData
        Your data
    labels : str
        Column with perturbation labels
    nt_class_name : str
        Name of NT cells
    split_by : str, optional
        Column with cell-type labels (if you have them)
    """
    
    if split_by is None:
        print("❌ No cell-type labels provided (split_by=None)")
        print("   → Harmonization NOT needed (only useful with multiple cell types)")
        return False
    
    print(f"Checking harmonization need using split_by='{split_by}'...\n")
    
    # Get all perturbations
    all_genes = [g for g in adata.obs[labels].unique() if g != nt_class_name]
    
    needs_harmonization = False
    
    for gene in all_genes[:5]:  # Check first 5 genes as examples
        nt_cells = adata.obs[labels] == nt_class_name
        gene_cells = adata.obs[labels] == gene
        
        # Cell-type distribution in NT
        nt_dist = adata.obs.loc[nt_cells, split_by].value_counts(normalize=True)
        # Cell-type distribution in perturbed
        gene_dist = adata.obs.loc[gene_cells, split_by].value_counts(normalize=True)
        
        # Align and compare
        all_types = sorted(set(nt_dist.index) | set(gene_dist.index))
        nt_props = [nt_dist.get(t, 0) for t in all_types]
        gene_props = [gene_dist.get(t, 0) for t in all_types]
        
        # Calculate difference (sum of absolute differences)
        diff = sum(abs(np.array(nt_props) - np.array(gene_props)))
        
        print(f"Gene: {gene}")
        print(f"  Cell-type distribution difference: {diff:.3f}")
        
        if diff > 0.3:  # Threshold: >30% difference
            print(f"  ⚠️  LARGE difference - harmonization recommended")
            needs_harmonization = True
        else:
            print(f"  ✓  Similar distributions")
        print()
    
    print("\n" + "="*60)
    if needs_harmonization:
        print("🔴 RECOMMENDATION: Implement harmonization")
        print("   Cell-type distributions differ significantly between")
        print("   NT and perturbed cells. This could bias DE results.")
    else:
        print("🟢 RECOMMENDATION: Harmonization NOT needed")
        print("   Cell-type distributions are similar enough.")
    print("="*60)
    
    return needs_harmonization


# Example usage:
# adata = sc.read_h5ad('your_data.h5ad')
# check_harmonization_need(adata, labels='gene', nt_class_name='NT', split_by='cell_type')
