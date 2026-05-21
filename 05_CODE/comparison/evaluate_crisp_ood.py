#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
REUSABLE = ROOT / "01_REUSABLE_ASSETS"
RUNTIME = ROOT / "02_RUNTIME_RESULTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=REUSABLE / "preprocessed_data" / "GBM_Universal_Perturbation_Ready.h5ad")
    parser.add_argument("--predicted", type=Path, required=True, help="AnnData with predicted OOD cells in X.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-patient", default=None)
    parser.add_argument("--target-drug", default=None)
    parser.add_argument("--output-md", type=Path, default=RUNTIME / "evaluation" / "legacy_nips" / "GBM_CRISP_OOD_metrics.md")
    parser.add_argument("--sinkhorn-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def sinkhorn_distance(x_pred: np.ndarray, x_true: np.ndarray, samples: int, seed: int) -> float:
    import torch
    from geomloss import SamplesLoss

    rng = np.random.default_rng(seed)
    if x_pred.shape[0] > samples:
        x_pred = x_pred[rng.choice(x_pred.shape[0], samples, replace=False)]
    if x_true.shape[0] > samples:
        x_true = x_true[rng.choice(x_true.shape[0], samples, replace=False)]
    loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")
    with torch.no_grad():
        xp = torch.as_tensor(x_pred, dtype=torch.float32)
        xt = torch.as_tensor(x_true, dtype=torch.float32)
        value = loss(xp, xt).detach().cpu().item()
    return float(value)


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.adata)
    pred = ad.read_h5ad(args.predicted)
    target = adata.uns.get("ood_split", {})
    patient = args.target_patient or target.get("target_patient")
    drug = args.target_drug or target.get("target_drug")
    if not patient or not drug:
        raise ValueError("Target patient/drug must be provided or present in adata.uns['ood_split']")
    key = f"{patient}|{drug}"
    genes = adata.uns["top50_DEGs"][key]
    gene_idx = adata.var_names.get_indexer(genes)
    if np.any(gene_idx < 0):
        missing = [g for g, i in zip(genes, gene_idx) if i < 0]
        raise ValueError(f"Missing Top50 genes in adata: {missing[:5]}")
    pred_idx = pred.var_names.get_indexer(genes)
    if np.any(pred_idx < 0):
        missing = [g for g, i in zip(genes, pred_idx) if i < 0]
        raise ValueError(f"Missing Top50 genes in predictions: {missing[:5]}")

    true_mask = (adata.obs["cell_type"].eq(patient) & adata.obs["perturbation"].eq(drug)).to_numpy()
    ctrl_mask = (adata.obs["cell_type"].eq(patient) & adata.obs["is_control"]).to_numpy()
    if true_mask.sum() == 0 or ctrl_mask.sum() == 0:
        raise ValueError(f"Missing true/control cells for {key}")

    x_true = dense(adata.X[true_mask][:, gene_idx])
    x_ctrl = dense(adata.X[ctrl_mask][:, gene_idx])
    x_pred = dense(pred.X[:, pred_idx])
    true_post = x_true.mean(axis=0)
    pred_post = x_pred.mean(axis=0)
    ctrl = x_ctrl.mean(axis=0)
    true_logfc = true_post - ctrl
    pred_logfc = pred_post - ctrl

    pr = stats.pearsonr(pred_logfc, true_logfc).statistic
    sp = stats.spearmanr(pred_logfc, true_logfc).statistic
    r2 = r2_score(true_post, pred_post)
    sink = sinkhorn_distance(x_pred, x_true, args.sinkhorn_samples, args.seed)
    direction = float(np.mean(np.sign(pred_logfc) == np.sign(true_logfc)) * 100.0)

    table = (
        "| Method | Target Covariate (Patient) | Target Drug | PrΔ DE (↑) | Sp DE (↑) | R² score DE (↑) | Sinkhorn DE (↓) | Direction Accuracy (%) (↑) |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| {args.method} | {patient} | {drug} | {pr:.3f} | {sp:.3f} | {r2:.3f} | {sink:.3f} | {direction:.1f}% |\n"
    )
    args.output_md.write_text(table)
    print(table)


if __name__ == "__main__":
    main()
