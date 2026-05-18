#!/usr/bin/env python
"""GBM 数据集预处理主入口：将 scGPT/MolFormer embedding 合并到原始数据。

此脚本运行于 **plknature** 环境（可读写 GBM h5ad 格式），
读取由 nature 环境产出的中间文件（scGPT embeddings, MolFormer embeddings），
合并到原始 GBM h5ad 并输出新的 h5ad 文件。

前置步骤（在 nature 环境中运行）：
  1. python scripts/encode_gbm_cells_scgpt.py     → GBM_scGPT_embeddings.h5ad
  2. python scripts/encode_gbm_drugs_molformer.py  → GBM_X_MolFormer.npy

用法：
  conda activate plknature
  python scripts/prepare_gbm_with_embeddings.py \
      --adata GBM_Universal_Perturbation_Ready.h5ad \
      --scgpt GBM_scGPT_embeddings.h5ad \
      --molformer GBM_X_MolFormer.npy \
      --output GBM_with_embeddings.h5ad \
      --drug-parquet GBM_molformer_drug_emb.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge scGPT/MolFormer embeddings into GBM AnnData"
    )
    parser.add_argument(
        "--adata",
        type=Path,
        default=ROOT / "GBM_Universal_Perturbation_Ready.h5ad",
        help="Original GBM h5ad (read-only, never modified).",
    )
    parser.add_argument(
        "--scgpt",
        type=Path,
        default=ROOT / "GBM_scGPT_embeddings.h5ad",
        help="scGPT output h5ad containing obsm X_scGPT, X_scGPT_ctrl, X_scGPT_pert.",
    )
    parser.add_argument(
        "--molformer",
        type=Path,
        default=ROOT / "GBM_X_MolFormer.npy",
        help="MolFormer per-cell embeddings (npy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "GBM_with_embeddings.h5ad",
        help="Output h5ad with all embeddings in obsm.",
    )
    parser.add_argument(
        "--drug-parquet",
        type=Path,
        default=ROOT / "GBM_molformer_drug_emb.parquet",
        help="Drug-level parquet to copy (already produced by encode_gbm_drugs_molformer.py).",
    )
    parser.add_argument(
        "--skip-scgpt",
        action="store_true",
        help="Skip scGPT embedding merge.",
    )
    parser.add_argument(
        "--skip-molformer",
        action="store_true",
        help="Skip MolFormer embedding merge.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        help="HDF5 compression for output.",
    )
    return parser.parse_args()


def merge_scgpt_embeddings(adata: AnnData, scgpt_path: Path) -> AnnData:
    """Merge scGPT embeddings from scGPT output h5ad into target adata."""
    if not scgpt_path.exists():
        raise FileNotFoundError(f"scGPT output not found: {scgpt_path}")

    scgpt_adata = sc.read_h5ad(scgpt_path, backed="r")
    try:
        expected_keys = ["X_scGPT", "X_scGPT_ctrl", "X_scGPT_pert"]
        missing = [k for k in expected_keys if k not in scgpt_adata.obsm]
        if missing:
            raise KeyError(f"Missing obsm keys in scGPT output: {missing}")

        for key in expected_keys:
            arr = scgpt_adata.obsm[key]
            if arr.shape[0] != adata.n_obs:
                raise ValueError(
                    f"scGPT embedding {key} has {arr.shape[0]} rows "
                    f"but adata has {adata.n_obs} cells"
                )
            adata.obsm[key] = arr.astype(np.float32, copy=True)

        print(f"  Merged scGPT embeddings: {expected_keys}")
        print(f"    X_scGPT shape: {adata.obsm['X_scGPT'].shape}")

        n_ctrl = int(adata.obs["is_control"].sum())
        ctrl_nonzero = int(np.count_nonzero(adata.obsm["X_scGPT_ctrl"]))
        pert_nonzero = int(np.count_nonzero(adata.obsm["X_scGPT_pert"]))
        print(f"    X_scGPT_ctrl nonzero: {ctrl_nonzero}")
        print(f"    X_scGPT_pert nonzero: {pert_nonzero}")
    finally:
        scgpt_adata.file.close()

    return adata


def merge_molformer_embeddings(adata: AnnData, molformer_path: Path) -> AnnData:
    """Merge MolFormer per-cell embeddings from npy into target adata."""
    if not molformer_path.exists():
        raise FileNotFoundError(f"MolFormer npy not found: {molformer_path}")

    cell_embs = np.load(molformer_path)
    if cell_embs.shape[0] != adata.n_obs:
        raise ValueError(
            f"MolFormer embeddings have {cell_embs.shape[0]} rows "
            f"but adata has {adata.n_obs} cells"
        )

    adata.obsm["X_MolFormer"] = cell_embs.astype(np.float32, copy=False)
    print(f"  Merged MolFormer embeddings: {cell_embs.shape}")
    return adata


def add_smiles_and_condition(adata: AnnData) -> AnnData:
    """Add SMILES, condition, and canonical_smiles columns to obs."""
    drug_smiles = adata.uns.get("drug_smiles", {})

    adata.obs["SMILES"] = adata.obs["perturbation"].map(
        lambda p: drug_smiles.get(str(p), "")
    ).astype(str)

    adata.obs["condition"] = np.where(
        adata.obs["is_control"], "control", "treated"
    )

    adata.obs["canonical_smiles"] = adata.obs["SMILES"].apply(
        lambda smi: Chem.CanonSmiles(str(smi)) if smi and not pd.isna(smi) else ""
    )

    print("  Added obs columns: SMILES, condition, canonical_smiles")
    return adata


def validate_output(adata: AnnData, args: argparse.Namespace) -> None:
    """Validate output integrity."""
    n_cells = adata.n_obs
    errors: list[str] = []

    if not args.skip_scgpt:
        for key in ["X_scGPT", "X_scGPT_ctrl", "X_scGPT_pert"]:
            if key not in adata.obsm:
                errors.append(f"Missing obsm key: {key}")
                continue
            arr = adata.obsm[key]
            if arr.shape[0] != n_cells:
                errors.append(f"{key}: expected {n_cells} rows, got {arr.shape[0]}")
            if arr.shape[1] != 512:
                errors.append(f"{key}: expected 512 dims, got {arr.shape[1]}")

            # Verify masks
            if key == "X_scGPT_ctrl":
                is_ctrl = adata.obs["is_control"].to_numpy(bool)
                pert_rows = arr[~is_ctrl]
                if not np.allclose(pert_rows, 0):
                    errors.append(f"{key}: perturbation rows should be zero")
            elif key == "X_scGPT_pert":
                is_ctrl = adata.obs["is_control"].to_numpy(bool)
                ctrl_rows = arr[is_ctrl]
                if not np.allclose(ctrl_rows, 0):
                    errors.append(f"{key}: control rows should be zero")

    if not args.skip_molformer:
        key = "X_MolFormer"
        if key not in adata.obsm:
            errors.append(f"Missing obsm key: {key}")
        else:
            arr = adata.obsm[key]
            if arr.shape[0] != n_cells:
                errors.append(f"{key}: expected {n_cells} rows, got {arr.shape[0]}")
            if arr.shape[1] != 768:
                errors.append(f"{key}: expected 768 dims, got {arr.shape[1]}")

    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("  Validation: OK")


def main() -> None:
    args = parse_args()

    adata_path = Path(args.adata).resolve()
    output_path = Path(args.output).resolve()
    scgpt_path = Path(args.scgpt).resolve()
    molformer_path = Path(args.molformer).resolve()

    if not adata_path.exists():
        raise FileNotFoundError(f"Input h5ad not found: {adata_path}")

    print(f"Loading original GBM data: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"  obsm keys before: {list(adata.obsm.keys())}")

    if not args.skip_scgpt:
        print("Merging scGPT embeddings ...")
        adata = merge_scgpt_embeddings(adata, scgpt_path)
    else:
        print("Skipping scGPT embeddings")

    if not args.skip_molformer:
        print("Merging MolFormer embeddings ...")
        adata = merge_molformer_embeddings(adata, molformer_path)
    else:
        print("Skipping MolFormer embeddings")

    print("Adding SMILES and condition columns ...")
    adata = add_smiles_and_condition(adata)

    print("Validating ...")
    validate_output(adata, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving: {output_path}")
    adata.write_h5ad(output_path, compression=args.compression)
    print(f"  obsm keys after: {list(adata.obsm.keys())}")
    print("Done.")


if __name__ == "__main__":
    main()
