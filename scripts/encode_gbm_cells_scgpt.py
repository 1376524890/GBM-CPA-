#!/usr/bin/env python
"""用 scGPT 对 GBM 数据集细胞编码，生成 512d embedding。

运行环境：conda activate nature

输出 obsm 键：
  - X_scGPT      : 全部细胞的 scGPT embedding (n_cells, 512)
  - X_scGPT_ctrl : 仅对照组保留 scGPT，扰动组置零
  - X_scGPT_pert : 仅扰动组保留 scGPT，对照组置零
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
from scgpt.tasks import embed_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = "/home/u2023312303/nature子刊/zyq/encoder/scGPT_blood"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode GBM cells with scGPT"
    )
    parser.add_argument(
        "--adata",
        type=Path,
        default=ROOT / "GBM_counts_for_scgpt.h5ad",
        help="Input h5ad with integer counts (compatible with nature env anndata).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "GBM_scGPT_embeddings.h5ad",
        help="Output h5ad with scGPT embeddings in obsm.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing scGPT pretrained model (args.json, best_model.pt, vocab.json).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for scGPT inference.",
    )
    parser.add_argument(
        "--control-col",
        default="is_control",
        help="obs column name for control flag (True=control).",
    )
    parser.add_argument(
        "--gene-col",
        default="index",
        choices=["index", "gene_name", "gene_ids"],
        help="How to match genes between data and model vocab. 'index' uses var_names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adata_path = Path(args.adata).resolve()
    output_path = Path(args.output).resolve()
    model_dir = Path(args.model_dir)

    if not adata_path.exists():
        raise FileNotFoundError(f"Input h5ad not found: {adata_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    for required_file in ["args.json", "best_model.pt", "vocab.json"]:
        if not (model_dir / required_file).exists():
            raise FileNotFoundError(
                f"Missing {required_file} in model directory {model_dir}"
            )

    print(f"Loading: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  {adata.n_obs} cells, {adata.n_vars} genes")

    if args.control_col not in adata.obs.columns:
        available = list(adata.obs.columns)
        raise KeyError(
            f"Control column '{args.control_col}' not found in obs. Available: {available}"
        )
    is_control = adata.obs[args.control_col].to_numpy(dtype=bool)
    n_ctrl = int(is_control.sum())
    n_pert = int((~is_control).sum())
    print(f"  control: {n_ctrl}, perturbation: {n_pert}")

    print(f"Running scGPT embed_data (model: {model_dir}) ...")
    sys.stdout.flush()

    adata_embed = embed_data(
        adata,
        model_dir=str(model_dir),
        gene_col=args.gene_col,
        batch_size=args.batch_size,
        return_new_adata=True,
    )

    embeddings = np.asarray(adata_embed.X, dtype=np.float32)
    print(f"  embedding shape: {embeddings.shape}")

    adata.obsm["X_scGPT"] = embeddings

    ctrl_emb = np.zeros_like(embeddings)
    ctrl_emb[is_control] = embeddings[is_control]
    adata.obsm["X_scGPT_ctrl"] = ctrl_emb

    pert_emb = np.zeros_like(embeddings)
    pert_emb[~is_control] = embeddings[~is_control]
    adata.obsm["X_scGPT_pert"] = pert_emb

    ctrl_nonzero = int(np.count_nonzero(ctrl_emb))
    pert_nonzero = int(np.count_nonzero(pert_emb))
    print(f"  X_scGPT_ctrl nonzero: {ctrl_nonzero} (expected: {n_ctrl * embeddings.shape[1]})")
    print(f"  X_scGPT_pert nonzero: {pert_nonzero} (expected: {n_pert * embeddings.shape[1]})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
