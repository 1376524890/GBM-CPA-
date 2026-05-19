#!/usr/bin/env python3
"""
Re-evaluate Counts-Space Predictions in Log1p Space
====================================================
CPA M0 and M4 legacy predictions are in counts space. This script converts
them to log1p space and recomputes evaluation metrics for fair comparison
with log1p-space methods (M1/M2/M3/M5/MLP/MeanShift).

Usage:
    python reevaluate_counts_predictions_log1p.py [--h5ad PATH] [--output-dir PATH]
"""
import argparse
import os
import json
import numpy as np
import anndata as ad
import scipy.sparse as sp
import torch
import warnings
warnings.filterwarnings("ignore")


def get_deg_genes(adata, group_name):
    if "rank_genes_groups_cov" in adata.uns and group_name in adata.uns["rank_genes_groups_cov"]:
        return adata.uns["rank_genes_groups_cov"][group_name]
    if "top50_DEGs" in adata.uns:
        t50 = adata.uns["top50_DEGs"]
        legacy_key = group_name.replace("_", "|")
        if legacy_key in t50:
            return t50[legacy_key]
    return None


def load_matrix_from_adata(a):
    """Load X from AnnData, converting to dense float64 array safely."""
    if sp.issparse(a.X):
        return a.X.toarray().astype(np.float64)
    return np.asarray(a.X, dtype=np.float64)


def compute_metrics(Y_true, Y_pred, Y_ctrl, deg_genes, var_names):
    """Compute per-group evaluation metrics."""
    # DEG indices
    deg_idx = [list(var_names).index(g) for g in deg_genes if g in var_names]

    results = {}

    # --- Full-gene mean-profile metrics ---
    mu_true = Y_true.mean(axis=0)
    mu_pred = Y_pred.mean(axis=0)
    mu_ctrl = Y_ctrl.mean(axis=0)

    results["pearson"] = float(np.corrcoef(mu_true, mu_pred)[0, 1])
    results["mse"] = float(np.mean((mu_true - mu_pred) ** 2))
    ss_res = np.sum((mu_true - mu_pred) ** 2)
    ss_tot = np.sum((mu_true - mu_true.mean()) ** 2)
    results["r2score"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    delta_true = mu_true - mu_ctrl
    delta_pred = mu_pred - mu_ctrl
    results["pearson_delta"] = float(np.corrcoef(delta_true, delta_pred)[0, 1])

    # --- DEG-subset mean-profile metrics ---
    if deg_idx:
        mu_true_de = mu_true[deg_idx]
        mu_pred_de = mu_pred[deg_idx]
        mu_ctrl_de = mu_ctrl[deg_idx]

        # pearson_de on DEG subset is typically not meaningful for small DEG sets
        results["pearson_de"] = float(np.corrcoef(mu_true_de, mu_pred_de)[0, 1]) if len(deg_idx) > 2 else 0.0
        results["mse_de"] = float(np.mean((mu_true_de - mu_pred_de) ** 2))
        ss_res_de = np.sum((mu_true_de - mu_pred_de) ** 2)
        ss_tot_de = np.sum((mu_true_de - mu_true_de.mean()) ** 2)
        results["r2score_de"] = float(1 - ss_res_de / ss_tot_de) if ss_tot_de > 0 else 0.0

        delta_true_de = mu_true_de - mu_ctrl_de
        delta_pred_de = mu_pred_de - mu_ctrl_de
        results["pearson_delta_de"] = float(np.corrcoef(delta_true_de, delta_pred_de)[0, 1])
    else:
        for k in ["pearson_de", "mse_de", "r2score_de", "pearson_delta_de"]:
            results[k] = 0.0

    # --- Sinkhorn DE ---
    try:
        import geomloss
        if deg_idx:
            Y_true_de = Y_true[:, deg_idx]
            Y_pred_de = Y_pred[:, deg_idx]
            Y_true_de = Y_true_de / (np.linalg.norm(Y_true_de, axis=1, keepdims=True) + 1e-8)
            Y_pred_de = Y_pred_de / (np.linalg.norm(Y_pred_de, axis=1, keepdims=True) + 1e-8)
            sinkhorn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
            results["sinkhorn_de"] = float(sinkhorn(
                torch.from_numpy(Y_true_de.astype(np.float32)),
                torch.from_numpy(Y_pred_de.astype(np.float32))).item())
        else:
            results["sinkhorn_de"] = float("nan")
    except ImportError:
        results["sinkhorn_de"] = float("nan")
        print("  geomloss not available, sinkhorn_de skipped")

    # --- Direction accuracy ---
    if deg_idx:
        delta_true_de_cell = Y_true[:, deg_idx] - Y_ctrl.mean(axis=0)[deg_idx]
        delta_pred_de_cell = Y_pred[:, deg_idx] - Y_ctrl.mean(axis=0)[deg_idx]
        sign_match = np.sign(delta_true_de_cell.mean(axis=0)) == np.sign(delta_pred_de_cell.mean(axis=0))
        results["direction_accuracy_de"] = float(sign_match.mean())
    else:
        results["direction_accuracy_de"] = 0.0

    results["n_treated"] = Y_true.shape[0]
    results["n_ctrl"] = Y_ctrl.shape[0]
    results["n_pred"] = Y_pred.shape[0]

    return results


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate counts-space predictions in log1p")
    parser.add_argument("--h5ad", default=None, help="Path to GBM_NIPS_Ready.h5ad")
    parser.add_argument("--output-dir", default="evaluation_results_unified_log1p",
                        help="Output directory for re-evaluated metrics")
    args = parser.parse_args()

    # Find h5ad
    h5ad_path = args.h5ad
    if h5ad_path is None:
        for p in ["../GBM_NIPS_Ready.h5ad", "GBM_NIPS_Ready.h5ad"]:
            if os.path.exists(p):
                h5ad_path = p
                break
    if h5ad_path is None or not os.path.exists(h5ad_path):
        print(f"ERROR: Cannot find GBM_NIPS_Ready.h5ad")
        return

    print(f"Loading: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)

    # Prediction files with known space info
    pred_files = {
        "CPA_M0": {"file": "../GBM_CPA_PW034_Panobinostat_pred.h5ad", "space": "counts"},
        "CPA_M4_MolFormer": {"file": "../GBM_CPA_MolFormer_PW034_Panobinostat_pred.h5ad", "space": "counts"},
        "CPA_M1_scGPT": {"file": "../GBM_CPA_scGPT_PW034_Panobinostat_pred.h5ad", "space": "log1p"},
        "CPA_M2_scGPT_ctrl": {"file": "../GBM_CPA_scGPT_ctrl_PW034_Panobinostat_pred.h5ad", "space": "log1p"},
        "CPA_M3_scGPT_pert": {"file": "../GBM_CPA_scGPT_pert_PW034_Panobinostat_pred.h5ad", "space": "log1p"},
        "CPA_M5_scGPT_MolFormer": {"file": "../GBM_CPA_scGPT_MolFormer_PW034_Panobinostat_pred.h5ad", "space": "log1p"},
        "MeanShiftBaseline": {"file": "../GBM_MeanShiftBaseline_PW034_Panobinostat_pred.h5ad", "space": "log1p"},
    }

    os.makedirs(args.output_dir, exist_ok=True)

    ood_group = "PW034_Panobinostat"
    deg_genes = get_deg_genes(adata, ood_group)
    if deg_genes is None:
        print(f"ERROR: No DEG genes found for {ood_group}")
        return
    print(f"DEG genes for {ood_group}: {len(deg_genes)}")

    # Y_true and Y_ctrl from reference
    treated_mask = adata.obs["cov_drug_name"] == ood_group
    ctrl_mask = adata.obs["cov_drug_name"] == "PW034_control"
    Y_true = load_matrix_from_adata(adata[treated_mask])
    Y_ctrl = load_matrix_from_adata(adata[ctrl_mask])
    print(f"Y_true (log1p): {Y_true.shape}, Y_ctrl (log1p): {Y_ctrl.shape}")

    results_summary = {}

    for method, info in pred_files.items():
        pred_path = info["file"]
        if not os.path.exists(pred_path):
            print(f"\n{method}: FILE NOT FOUND ({pred_path})")
            continue

        print(f"\n{'=' * 60}")
        print(f"{method} (legacy space: {info['space']})")
        print(f"{'=' * 60}")

        pred = ad.read_h5ad(pred_path)
        Y_pred_raw = load_matrix_from_adata(pred)

        if info["space"] == "counts":
            Y_pred_eval = np.log1p(np.maximum(Y_pred_raw, 0))
            print(f"  Converted: counts -> log1p")
        else:
            Y_pred_eval = Y_pred_raw
            print(f"  Already in log1p space, using as-is")

        print(f"  Y_pred shape: {Y_pred_eval.shape}")
        print(f"  Y_pred value range: [{Y_pred_eval.min():.4f}, {Y_pred_eval.max():.4f}]")

        metrics = compute_metrics(Y_true, Y_pred_eval, Y_ctrl, deg_genes, adata.var_names)

        # Print key metrics
        for k in ["pearson_delta_de", "sinkhorn_de", "pearson", "pearson_delta", "mse", "direction_accuracy_de"]:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.6f}")

        results_summary[method] = {
            "method": method,
            "legacy_space": info["space"],
            "eval_space": "log1p",
            "macro_avg": metrics,
            "per_group": {ood_group: metrics},
        }

        # Save individual result
        out_path = os.path.join(args.output_dir, f"{method}_ood_metrics_log1p.json")
        with open(out_path, "w") as f:
            json.dump(results_summary[method], f, indent=2)
        print(f"  Saved: {out_path}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("UNIFIED LOG1P EVALUATION SUMMARY (OOD: PW034_Panobinostat)")
    print(f"{'=' * 80}")
    print(f"{'Method':<25s} {'LegacySpace':<12s} {'PrDelta_DE':<12s} {'Sinkhorn_DE':<12s} {'Pearson':<12s} {'DirAcc_DE':<12s}")
    print("-" * 80)
    for method in sorted(results_summary.keys()):
        m = results_summary[method]["macro_avg"]
        ls = results_summary[method]["legacy_space"]
        print(f"{method:<25s} {ls:<12s} {m.get('pearson_delta_de', 0):<12.4f} "
              f"{m.get('sinkhorn_de', float('nan')):<12.6f} "
              f"{m.get('pearson', 0):<12.4f} "
              f"{m.get('direction_accuracy_de', 0):<12.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "unified_log1p_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nFull summary saved: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
