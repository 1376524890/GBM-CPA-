#!/usr/bin/env python
"""Multi-group NIPS-style evaluation for GBM perturbation prediction.

Follows the unified evaluation protocol from LYZ_LZX_NIPS_METRICS.md:
  1. Iterate over valid (cell_type, drug) groups
  2. Build mean profiles: y=mean(Y_true), p=mean(Y_pred), c=mean(Y_ctrl)
  3. Compute per-group metrics (r2score, pearson, mse, delta, sinkhorn_de)
  4. Unweighted macro average over valid groups

Usage:
  conda activate plknature
  python 00_DELIVERY_CURRENT/code/cpa_nips/evaluate_nips_gbm.py --predicted 00_DELIVERY_CURRENT/predictions/GBM_CPA_NIPS_PW034_Panobinostat_pred.h5ad --method CPA_NIPS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import mean_squared_error as mse

from nips_aliases import control_mask, ensure_nips_aliases

ROOT = Path(__file__).resolve().parents[3]
DELIVERY = ROOT / "00_DELIVERY_CURRENT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=DELIVERY / "dataset" / "GBM_NIPS_Ready.h5ad")
    parser.add_argument("--predicted", type=Path, required=True,
                        help="AnnData with predicted expression for all OOD cells, or a single-group prediction.")
    parser.add_argument("--method", required=True, help="Method label for output.")
    parser.add_argument("--setting", default="ood", choices=["ood", "valid", "all"],
                        help="Which split to evaluate: 'ood' for OOD, 'valid' for IID, 'all' for both")
    parser.add_argument("--output-dir", type=Path, default=DELIVERY / "evaluation" / "nips")
    parser.add_argument("--min-treated", type=int, default=5)
    parser.add_argument("--min-ctrl", type=int, default=5)
    parser.add_argument("--min-deg", type=int, default=2)
    parser.add_argument("--sinkhorn-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return max(r2, 0.0)


def sinkhorn_distance(x_pred, x_true, samples=512, seed=7):
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
        return float(loss(xp, xt).detach().cpu().item())


def calc_metrics_nips(yt_m, yp_m, ctrl_m, y_true, preds, idx_de):
    """Compute per-group metrics following CRISP eval.py::calc_metrics."""
    metrics = {}
    yt_de_m = yt_m[idx_de].copy()
    yp_de_m = yp_m[idx_de].copy()

    # Zero-sum guard (from CRISP)
    if yt_de_m.sum() == 0:
        yt_de_m[0] += 1e-6
    if yp_de_m.sum() == 0:
        yp_de_m[0] += 1e-6

    metrics["r2score"] = compute_r2(yt_m, yp_m)
    metrics["r2score_de"] = compute_r2(yt_m[idx_de], yp_m[idx_de])
    metrics["pearson"] = max(stats.pearsonr(yt_m, yp_m).statistic, 0) if not np.isnan(stats.pearsonr(yt_m, yp_m).statistic) else 0.0
    metrics["pearson_de"] = max(stats.pearsonr(yt_de_m, yp_de_m).statistic, 0) if not np.isnan(stats.pearsonr(yt_de_m, yp_de_m).statistic) else 0.0
    metrics["mse"] = float(mse(yt_m, yp_m))
    metrics["mse_de"] = float(mse(yt_m[idx_de], yp_m[idx_de]))

    # Delta metrics
    true_delta = yt_m - ctrl_m
    pred_delta = yp_m - ctrl_m
    metrics["pearson_delta"] = max(stats.pearsonr(true_delta, pred_delta).statistic, 0) if not np.isnan(stats.pearsonr(true_delta, pred_delta).statistic) else 0.0

    true_delta_de = yt_de_m - ctrl_m[idx_de]
    pred_delta_de = yp_de_m - ctrl_m[idx_de]
    metrics["pearson_delta_de"] = max(stats.pearsonr(true_delta_de, pred_delta_de).statistic, 0) if not np.isnan(stats.pearsonr(true_delta_de, pred_delta_de).statistic) else 0.0

    # Sinkhorn (cell-level)
    if preds[:, idx_de].sum() == 0 and y_true[:, idx_de].sum() == 0:
        metrics["sinkhorn_de"] = 0.0
    else:
        metrics["sinkhorn_de"] = sinkhorn_distance(preds[:, idx_de], y_true[:, idx_de])

    # Handle NaN
    for k in ["pearson", "pearson_de", "pearson_delta", "pearson_delta_de"]:
        if np.isnan(metrics[k]):
            metrics[k] = 0.0

    return metrics


def evaluate_predictions(adata_path, pred_path, method, setting, args):
    """Evaluate predictions following NIPS protocol."""
    adata = ad.read_h5ad(adata_path, backed="r")
    pred = ad.read_h5ad(pred_path)
    ensure_nips_aliases(adata, add_legacy_aliases=True)
    ensure_nips_aliases(pred, add_legacy_aliases=True, require_degs=False)

    # Get the groups to evaluate
    if setting == "ood":
        split_mask = adata.obs["split"].astype(str) == "ood"
    elif setting == "valid":
        split_mask = adata.obs["split"].astype(str) == "valid"
    else:
        split_mask = np.ones(adata.n_obs, dtype=bool)

    # Build valid group list
    obs_split = adata.obs.loc[split_mask]
    cov_drug_groups = obs_split["cov_drug_name"].unique()

    # Get DEGs
    deg_dict = adata.uns.get("rank_genes_groups_cov", adata.uns.get("top50_DEGs", {}))
    var_names = adata.var_names

    eval_scores = {}
    pred_means = {}
    valid_groups = 0
    skipped = {"treated_too_few": 0, "dmso_control": 0, "deg_missing": 0,
               "deg_too_few": 0, "ctrl_too_few": 0}

    for group in cov_drug_groups:
        # Find cells in this group
        # Treated cells: must be in the evaluation split
        cell_type_str = str(group).split("_")[0]
        drug_str = "_".join(str(group).split("_")[1:])
        eval_split_mask_np = (adata.obs["split"].astype(str) == obs_split.loc[split_mask].iloc[0]["split"]).to_numpy() if setting != "all" else np.ones(adata.n_obs, dtype=bool)
        treated_mask_np = eval_split_mask_np & (adata.obs["cov_drug_name"].astype(str) == str(group)).to_numpy() & (~control_mask(adata.obs))
        # Control cells: same cell_type, is_control. Try evaluation split first, then any split
        ctrl_mask_np = eval_split_mask_np & (adata.obs["cell_type"].astype(str) == cell_type_str).to_numpy() & control_mask(adata.obs)
        if ctrl_mask_np.sum() < args.min_ctrl:
            ctrl_mask_np = (adata.obs["cell_type"].astype(str) == cell_type_str).to_numpy() & control_mask(adata.obs)

        n_treated = int(treated_mask_np.sum())
        n_ctrl = int(ctrl_mask_np.sum())

        # Filter 1: treated count > 5
        if n_treated <= args.min_treated:
            skipped["treated_too_few"] += 1
            continue

        # Filter 2: no dmso/control
        if "dmso" in str(group).lower() or "control" in str(group).lower():
            skipped["dmso_control"] += 1
            continue

        # Filter 3: DEG entry exists (try both _ and | separators)
        deg_key_underscore = str(group)
        deg_key_pipe = str(group).replace("_", "|", 1)  # Only first underscore
        if deg_key_underscore in deg_dict:
            deg_genes = deg_dict[deg_key_underscore]
        elif deg_key_pipe in deg_dict:
            deg_genes = deg_dict[deg_key_pipe]
        else:
            skipped["deg_missing"] += 1
            continue
        idx_de = np.array([i for i, g in enumerate(var_names) if g in deg_genes])
        if len(idx_de) < args.min_deg:
            skipped["deg_too_few"] += 1
            continue

        # Filter 4: control count >= 5
        if n_ctrl < args.min_ctrl:
            skipped["ctrl_too_few"] += 1
            continue

        # Build mean profiles
        Y_true = dense(adata.X[treated_mask_np])
        Y_ctrl = dense(adata.X[ctrl_mask_np])

        # Get predictions for this group
        # For CPA predictions: prefer NIPS original fields, then legacy aliases.
        if "cell_type" in pred.obs.columns and "condition" in pred.obs.columns:
            pred_mask_np = (pred.obs["cell_type"].astype(str) == cell_type_str).to_numpy() & (pred.obs["condition"].astype(str) == drug_str).to_numpy()
        elif "cell_type" in pred.obs.columns and "perturbation" in pred.obs.columns:
            pred_mask_np = (pred.obs["cell_type"].astype(str) == cell_type_str).to_numpy() & (pred.obs["perturbation"].astype(str) == drug_str).to_numpy()
        elif "covariate_patient" in pred.obs.columns and "perturbation" in pred.obs.columns:
            pred_mask_np = (pred.obs["covariate_patient"].astype(str) == cell_type_str).to_numpy() & (pred.obs["perturbation"].astype(str) == drug_str).to_numpy()
        else:
            pred_mask_np = np.ones(pred.n_obs, dtype=bool)

        if pred_mask_np.sum() == 0:
            skipped["ctrl_too_few"] += 1
            continue

        Y_pred = dense(pred.X[pred_mask_np])

        # Compute mean profiles
        y = Y_true.mean(axis=0)
        p = Y_pred.mean(axis=0)
        c = Y_ctrl.mean(axis=0)

        # Compute metrics
        metrics = calc_metrics_nips(y, p, c, Y_true, Y_pred, idx_de)
        metrics["n_treated"] = n_treated
        metrics["n_ctrl"] = n_ctrl
        metrics["n_pred"] = int(pred_mask_np.sum())
        eval_scores[str(group)] = metrics
        pred_means[str(group)] = {"true": y.tolist(), "pred": p.tolist(), "ctrl": c.tolist()}
        valid_groups += 1

    # Macro average
    if not eval_scores:
        print(f"No valid groups found! Skipped: {skipped}")
        return None, None, None

    metric_names = ["r2score", "r2score_de", "pearson", "pearson_de", "mse", "mse_de",
                    "pearson_delta", "pearson_delta_de", "sinkhorn_de"]
    macro_avg = {}
    for k in metric_names:
        vals = [eval_scores[g][k] for g in eval_scores]
        macro_avg[k] = float(np.mean(vals))

    macro_avg["n_valid_groups"] = valid_groups
    macro_avg["skipped"] = skipped

    return macro_avg, eval_scores, pred_means


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {args.method} on {args.setting} split...")
    macro_avg, eval_scores, pred_means = evaluate_predictions(
        args.adata, args.predicted, args.method, args.setting, args)

    if macro_avg is None:
        print("Evaluation failed - no valid groups.")
        sys.exit(1)

    # Save results
    result = {
        "method": args.method,
        "setting": args.setting,
        "macro_avg": macro_avg,
        "per_group": eval_scores,
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_json = args.output_dir / f"{args.method}_{args.setting}_metrics.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, cls=NpEncoder))
    print(f"Saved: {out_json}")

    # Print table
    print(f"\n{'='*80}")
    print(f"Method: {args.method} | Setting: {args.setting} | Groups: {macro_avg['n_valid_groups']}")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Value':>10}")
    print(f"{'-'*35}")
    for k in ["r2score", "r2score_de", "pearson", "pearson_de", "mse", "mse_de",
              "pearson_delta", "pearson_delta_de", "sinkhorn_de"]:
        print(f"{k:<25} {macro_avg[k]:>10.4f}")
    print(f"\nSkipped: {macro_avg['skipped']}")


if __name__ == "__main__":
    main()
