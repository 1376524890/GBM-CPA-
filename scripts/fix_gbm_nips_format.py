#!/usr/bin/env python
"""Add NIPS-protocol columns to GBM preprocessed data.

Converts GBM_with_embeddings.h5ad to match the NIPS data format required by
the unified evaluation protocol (LYZ_LZX_NIPS_METRICS.md).

Key additions:
  - cov_drug_name: cell_type + "_" + perturbation (group key for evaluation)
  - neg_control: 1 for control, 0 for treated (NIPS control definition)
  - Fix condition column: use drug names instead of "control"/"treated"
  - Ensure obsm keys match NIPS: X_scGPT, X_MolFormer

Usage:
  conda activate plknature
  python scripts/fix_gbm_nips_format.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "GBM_with_embeddings.h5ad")
    parser.add_argument("--output", type=Path, default=ROOT / "GBM_NIPS_Ready.h5ad")
    parser.add_argument("--compression", default="gzip")
    return parser.parse_args()


def add_nips_columns(adata: AnnData) -> AnnData:
    """Add NIPS-required columns to GBM data."""

    # 1. cov_drug_name: cell_type + "_" + perturbation
    adata.obs["cov_drug_name"] = (
        adata.obs["cell_type"].astype(str) + "_" + adata.obs["perturbation"].astype(str)
    )
    n_groups = adata.obs["cov_drug_name"].nunique()
    print(f"  cov_drug_name: {n_groups} unique groups")

    # 2. neg_control: 1 for control, 0 for treated (NIPS convention)
    adata.obs["neg_control"] = adata.obs["is_control"].astype(int)
    print(f"  neg_control: {adata.obs['neg_control'].value_counts().to_dict()}")

    # 3. Fix condition: use drug names instead of "control"/"treated"
    #    Keep original "condition" as "is_treated" for backward compat
    adata.obs["is_treated"] = adata.obs["condition"].copy()
    #    New condition = drug name (or "control" for control cells)
    adata.obs["condition"] = np.where(
        adata.obs["is_control"],
        "control",
        adata.obs["perturbation"].astype(str)
    )
    print(f"  condition (fixed): {sorted(adata.obs['condition'].unique())}")

    # 4. Verify SMILES column exists
    if "SMILES" not in adata.obs.columns:
        drug_smiles = adata.uns.get("drug_smiles", {})
        adata.obs["SMILES"] = adata.obs["perturbation"].map(
            lambda p: drug_smiles.get(str(p), "")
        ).astype(str)
        print("  SMILES: added from uns['drug_smiles']")

    # 5. Verify obsm keys
    for key in ["X_scGPT", "X_MolFormer"]:
        if key in adata.obsm:
            print(f"  obsm['{key}']: {adata.obsm[key].shape}")
        else:
            print(f"  WARNING: obsm['{key}'] MISSING!")

    return adata


def validate_output(adata: AnnData) -> None:
    """Validate NIPS compatibility."""
    required_obs = ["cov_drug_name", "neg_control", "condition", "SMILES",
                    "cell_type", "perturbation", "split", "is_control"]
    missing = [c for c in required_obs if c not in adata.obs.columns]
    if missing:
        print(f"ERROR: Missing obs columns: {missing}")
        sys.exit(1)

    # Check neg_control values
    nc_vals = adata.obs["neg_control"].unique()
    if not set(nc_vals).issubset({0, 1}):
        print(f"ERROR: neg_control has invalid values: {nc_vals}")
        sys.exit(1)

    # Check cov_drug_name format
    sample_groups = adata.obs["cov_drug_name"].iloc[:5].tolist()
    for g in sample_groups:
        if "_" not in g:
            print(f"ERROR: cov_drug_name missing underscore: {g}")
            sys.exit(1)

    # Check obsm
    for key in ["X_scGPT", "X_MolFormer"]:
        if key not in adata.obsm:
            print(f"ERROR: obsm['{key}'] missing!")
            sys.exit(1)

    print("  Validation: OK")
    print(f"  Final shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")


def main() -> None:
    args = parse_args()

    print(f"Loading: {args.input}")
    adata = sc.read_h5ad(args.input)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    print("Adding NIPS columns...")
    adata = add_nips_columns(adata)

    print("Validating...")
    validate_output(adata)

    print(f"Saving: {args.output}")
    adata.write_h5ad(args.output, compression=args.compression)
    print("Done.")


if __name__ == "__main__":
    main()
