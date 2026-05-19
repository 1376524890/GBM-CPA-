#!/usr/bin/env python3
"""
GBM NIPS-Ready Dataset - Minimal Loading Example
================================================
Demonstrates how to load the GBM dataset and prepare data for perturbation
prediction evaluation. Run this script to verify your environment is correctly
set up before running any baseline methods.

Usage:
    python load_gbm_example.py
"""

import numpy as np
import pandas as pd
import anndata as ad
import warnings
warnings.filterwarnings("ignore")

BASE_PATH = "/home/u2023312303/nature子刊/裴立昆实验"
H5AD_PATH = f"{BASE_PATH}/GBM_NIPS_Ready.h5ad"


def main():
    # =========================================================================
    # 1. Load the dataset
    # =========================================================================
    print("=" * 70)
    print("STEP 1: Load GBM_NIPS_Ready.h5ad")
    print("=" * 70)
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"  Loaded: {adata.shape[0]:,} cells x {adata.shape[1]} genes")

    # =========================================================================
    # 2. Print basic structure
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Basic Structure")
    print("=" * 70)
    print(f"  .X: {type(adata.X).__name__}, dtype={adata.X.dtype}")
    print(f"       (log1p-normalized expression)")
    print(f"  .layers: {list(adata.layers.keys())}")
    if "counts" in adata.layers:
        c = adata.layers["counts"]
        print(f"    counts: {type(c).__name__}, dtype={c.dtype}")
        print(f"    (raw UMI counts, non-negative integers)")
    print(f"  .obsm keys: {list(adata.obsm.keys())}")
    print(f"  .uns keys:  {list(adata.uns.keys())}")
    print(f"  .obs columns ({len(adata.obs.columns)}):")
    for col in adata.obs.columns:
        print(f"    - {col}: {adata.obs[col].dtype}")

    # =========================================================================
    # 3. Check required fields exist
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Verify Required Fields")
    print("=" * 70)

    required_obs = [
        "perturbation", "dosage", "covariate_patient",
        "split", "neg_control", "condition",
        "cov_drug_name", "cell_type", "SMILES", "canonical_smiles",
        "is_control", "is_treated"
    ]
    required_obsm = [
        "X_scGPT", "X_scGPT_ctrl", "X_scGPT_pert", "X_MolFormer"
    ]

    all_ok = True
    for field in required_obs:
        ok = field in adata.obs.columns
        if not ok:
            print(f"  ERROR: obs.{field} MISSING")
            all_ok = False
    for field in required_obsm:
        ok = field in adata.obsm
        if not ok:
            print(f"  ERROR: obsm['{field}'] MISSING")
            all_ok = False
    if all_ok:
        print("  All required fields present.")

    # =========================================================================
    # 4. Read counts, X_scGPT, X_MolFormer for example cells
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Access Key Matrices")
    print("=" * 70)

    # Log-normalized expression
    X_log1p = adata.X
    print(f"  X (log1p): shape={X_log1p.shape}, dtype={X_log1p.dtype}")

    # Raw UMI counts
    counts = adata.layers["counts"]
    print(f"  counts:    shape={counts.shape}, dtype={counts.dtype}")
    print(f"             min={counts.data.min()}, max={counts.data.max()}")

    # scGPT cell embeddings
    X_scGPT = adata.obsm["X_scGPT"]
    print(f"  X_scGPT:   shape={X_scGPT.shape}, dtype={X_scGPT.dtype}")
    print(f"             min={X_scGPT.min():.4f}, max={X_scGPT.max():.4f}")

    # MolFormer drug embeddings
    X_MolFormer = adata.obsm["X_MolFormer"]
    print(f"  X_MolFormer: shape={X_MolFormer.shape}, dtype={X_MolFormer.dtype}")

    # =========================================================================
    # 5. Select OOD group
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Select OOD Target Group for Evaluation")
    print("=" * 70)

    ood_target = "PW034_Panobinostat"

    # Find the corresponding control group
    patient = ood_target.split("_")[0]
    ctrl_group = f"{patient}_control"

    print(f"  OOD target: {ood_target}")
    print(f"  Matched control group: {ctrl_group}")

    # Get treated cells (OOD split)
    treated_mask = (adata.obs["cov_drug_name"] == ood_target) & (adata.obs["split"] == "ood")
    n_treated = treated_mask.sum()
    print(f"  Treated cells (ood split): {n_treated}")

    # Get control cells (PW034_control, all splits or specific split)
    # Typically use train/valid control cells as matched controls
    ctrl_mask = (adata.obs["cov_drug_name"] == ctrl_group)
    n_ctrl = ctrl_mask.sum()
    print(f"  Control cells (all splits): {n_ctrl}")

    # Split control by split
    for s in ["train", "valid", "ood"]:
        n = ((adata.obs["cov_drug_name"] == ctrl_group) & (adata.obs["split"] == s)).sum()
        print(f"    {s}: {n}")

    # =========================================================================
    # 6. Construct Y_true, Y_ctrl, and group info
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Construct Y_true, Y_ctrl, DEG genes")
    print("=" * 70)

    # Y_true: log1p expression of OOD treated cells
    Y_true = adata[treated_mask].X.toarray() if hasattr(adata[treated_mask].X, 'toarray') else adata[treated_mask].X
    if hasattr(Y_true, 'toarray'):
        Y_true = Y_true.toarray()
    print(f"  Y_true shape: {Y_true.shape}")

    # Y_ctrl: log1p expression of matched control cells
    Y_ctrl = adata[ctrl_mask].X.toarray() if hasattr(adata[ctrl_mask].X, 'toarray') else adata[ctrl_mask].X
    if hasattr(Y_ctrl, 'toarray'):
        Y_ctrl = Y_ctrl.toarray()
    print(f"  Y_ctrl shape: {Y_ctrl.shape}")

    # DEG genes for this group
    # NOTE: DEG keys use '|' separator, while cov_drug_name uses '_'
    deg_key = ood_target.replace("_", "|")
    deg_dict = adata.uns["top50_DEGs"]
    if deg_key in deg_dict:
        deg_genes = list(deg_dict[deg_key])
        print(f"  DEG genes: {len(deg_genes)} (key: '{deg_key}')")
        print(f"  First 5 DEGs: {deg_genes[:5]}")
    else:
        print(f"  WARNING: {deg_key} not found in top50_DEGs")
        deg_genes = []

    # =========================================================================
    # 7. Verify var_names consistency
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Gene Order Verification")
    print("=" * 70)
    print(f"  var_names count: {len(adata.var_names)}")
    print(f"  var_names unique: {adata.var_names.is_unique}")
    print(f"  First 10 genes: {list(adata.var_names[:10])}")
    print(f"  WARNING: All prediction matrices MUST use this exact gene order!")

    # =========================================================================
    # 8. Summary of the prediction evaluation protocol
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Evaluation Protocol Summary")
    print("=" * 70)
    print(f"""
    For each valid cov_drug_name group, construct:

    predictions = {{
        group_name: {{
            "Y_true": np.ndarray,   # treated cell log1p expression
            "Y_pred": np.ndarray,   # model-predicted expression
            "Y_ctrl": np.ndarray,   # control cell log1p expression
        }}
    }}

    Rules:
      - group_name MUST come from adata.obs['cov_drug_name']
      - Y_true columns MUST match adata.var_names order exactly
      - Y_pred columns MUST match adata.var_names order exactly
      - Y_ctrl is the matched {patient}_control group
      - Mean-profile metrics: compute mean over cells, then metric
      - sinkhorn_de: use DEG-subset cell-level matrices
      - Final score = unweighted macro average across valid groups
    """)

    print("Done. Dataset is ready for baseline method development.")


if __name__ == "__main__":
    main()
