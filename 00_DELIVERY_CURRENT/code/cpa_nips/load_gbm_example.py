#!/usr/bin/env python3
"""
GBM NIPS-Ready Dataset - Minimal Loading Example
================================================
Demonstrates how to load the unified GBM_NIPS_Ready.h5ad and prepare
data for perturbation prediction evaluation.

Usage:
    python load_gbm_example.py
    python load_gbm_example.py --h5ad /path/to/GBM_NIPS_Ready.h5ad
    python load_gbm_example.py --group PW030_Panobinostat
    python load_gbm_example.py --prediction-space counts
    python load_gbm_example.py --dense-limit 1000
"""
import argparse
import os
import sys
import numpy as np
import anndata as ad
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEFAULT_H5AD = os.path.join(ROOT, "00_DELIVERY_CURRENT", "dataset", "GBM_NIPS_Ready.h5ad")


def get_deg_genes(adata, group_name):
    """Get DEG genes for a group, trying rank_genes_groups_cov first."""
    if "rank_genes_groups_cov" in adata.uns:
        rg = adata.uns["rank_genes_groups_cov"]
        if group_name in rg:
            return list(rg[group_name])
    if "top50_DEGs" in adata.uns:
        t50 = adata.uns["top50_DEGs"]
        legacy_key = group_name.replace("_", "|")
        if legacy_key in t50:
            return list(t50[legacy_key])
    return []


def get_matched_control_mask(adata, group_name):
    """Return boolean mask for the matched control cells of a treated group."""
    if "control" in group_name.lower():
        raise ValueError(f"{group_name} is already a control group")
    patient = group_name.split("_")[0]
    ctrl_group = f"{patient}_control"
    if ctrl_group not in adata.obs["cov_drug_name"].unique():
        raise ValueError(f"No matched control group {ctrl_group} found")
    return adata.obs["cov_drug_name"] == ctrl_group


def normalize_prediction_to_eval_space(Y_pred, prediction_space):
    """Convert prediction to log1p evaluation space if needed."""
    if prediction_space == "counts":
        Y_pred = np.maximum(Y_pred, 0)
        return np.log1p(Y_pred)
    elif prediction_space == "log1p":
        return np.asarray(Y_pred, dtype=np.float32)
    else:
        raise ValueError(f"Unknown prediction_space: {prediction_space}")


def check_gene_order(pred_var_names, ref_var_names):
    """Verify prediction gene order matches reference."""
    if pred_var_names is None:
        return False, "pred var_names is None"
    if len(pred_var_names) != len(ref_var_names):
        return False, f"length mismatch: {len(pred_var_names)} vs {len(ref_var_names)}"
    if list(pred_var_names) != list(ref_var_names):
        return False, "order mismatch"
    return True, "OK"


def detect_h5ad_path():
    """Auto-detect GBM_NIPS_Ready.h5ad path."""
    candidates = [
        DEFAULT_H5AD,
        os.path.join("00_DELIVERY_CURRENT", "dataset", "GBM_NIPS_Ready.h5ad"),
        "GBM_NIPS_Ready.h5ad",
    ]
    # Also check cwd parent
    cwd_parent = os.path.join(os.getcwd(), "00_DELIVERY_CURRENT", "dataset", "GBM_NIPS_Ready.h5ad")
    candidates.append(cwd_parent)
    cwd_parent2 = os.path.join(os.path.dirname(os.getcwd()), "00_DELIVERY_CURRENT", "dataset", "GBM_NIPS_Ready.h5ad")
    candidates.append(cwd_parent2)

    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="GBM dataset loading example")
    parser.add_argument("--h5ad", default=None, help="Path to GBM_NIPS_Ready.h5ad (auto-detect if not given)")
    parser.add_argument("--group", default="PW034_Panobinostat", help="Target cov_drug_name group")
    parser.add_argument("--prediction-space", default="log1p", choices=["log1p", "counts"],
                        help="Expected prediction output space")
    parser.add_argument("--dense-limit", type=int, default=5000,
                        help="Max cells before warning about dense conversion")
    args = parser.parse_args()

    # Find h5ad
    h5ad_path = args.h5ad or detect_h5ad_path()
    if h5ad_path is None:
        print("ERROR: Cannot find GBM_NIPS_Ready.h5ad. Use --h5ad to specify path.")
        sys.exit(1)

    # =========================================================================
    # 1. Load
    # =========================================================================
    print("=" * 70)
    print("STEP 1: Load GBM_NIPS_Ready.h5ad")
    print("=" * 70)
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Path: {h5ad_path}")
    print(f"  Cells: {adata.n_obs:,}  Genes: {adata.n_vars:,}")

    # =========================================================================
    # 2. Print structure
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Data Structure")
    print("=" * 70)

    # Sparse stats
    if sp.issparse(adata.X):
        nnz = adata.X.nnz
        total = adata.n_obs * adata.n_vars
        density = nnz / total
        print(f"  .X:            {type(adata.X).__name__}, dtype={adata.X.dtype}")
        print(f"                 nnz={nnz:,}, density={density:.4f}")
        print(f"                 (log1p-normalized; nonzero_min={adata.X.data.min():.4f}, "
              f"nonzero_max={adata.X.data.max():.4f})")
        print(f"                 implicit_zero_ratio={1-density:.4f}")

    if "counts" in adata.layers:
        c = adata.layers["counts"]
        if sp.issparse(c):
            nnzc = c.nnz
            totalc = c.shape[0] * c.shape[1]
            densityc = nnzc / totalc
            print(f"  counts:        {type(c).__name__}, dtype={c.dtype}")
            print(f"                 nnz={nnzc:,}, density={densityc:.4f}")
            print(f"                 implicit_zero_ratio={1-densityc:.4f}")

    print(f"  obs keys ({len(adata.obs.columns)}): {list(adata.obs.columns)}")
    print(f"  obsm keys ({len(adata.obsm)}):      {list(adata.obsm.keys())}")
    print(f"  uns keys ({len(adata.uns)}):        {list(adata.uns.keys())}")
    print(f"  layers keys:  {list(adata.layers.keys())}")

    # =========================================================================
    # 3. Split and condition distributions
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Split & Condition Distribution")
    print("=" * 70)
    print("  Split:")
    for s in ["train", "valid", "ood"]:
        n = (adata.obs["split"] == s).sum()
        n_ctrl = ((adata.obs["split"] == s) & (adata.obs["neg_control"] == 1)).sum()
        n_trt = ((adata.obs["split"] == s) & (adata.obs["neg_control"] == 0)).sum()
        print(f"    {s}: {n:,d} cells (control={n_ctrl:,d}, treated={n_trt:,d})")

    print("  Condition:")
    for cond, n in adata.obs["condition"].value_counts().items():
        print(f"    {cond}: {n:,d}")

    # =========================================================================
    # 4. Embedding alias check
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Embedding Alias Consistency Check")
    print("=" * 70)
    for (alias, orig) in [("XscGPT", "X_scGPT"), ("XMolFormer", "X_MolFormer")]:
        if alias in adata.obsm and orig in adata.obsm:
            match = np.allclose(adata.obsm[alias], adata.obsm[orig], atol=1e-6)
            print(f"  {alias} == {orig}: {'PASS' if match else 'FAIL'}")
            print(f"    shape={adata.obsm[alias].shape}, dtype={adata.obsm[alias].dtype}")
        elif alias in adata.obsm:
            print(f"  {alias} exists, {orig} MISSING (unexpected)")
        else:
            print(f"  {alias} MISSING")

    # =========================================================================
    # 5. DEG alias check
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: DEG Dictionary Alias Check")
    print("=" * 70)
    for deg_key_name in ["rank_genes_groups_cov", "top50_DEGs"]:
        if deg_key_name in adata.uns:
            d = adata.uns[deg_key_name]
            print(f"  {deg_key_name}: {len(d)} entries")
        else:
            print(f"  {deg_key_name}: MISSING")

    if "rank_genes_groups_cov" in adata.uns and "top50_DEGs" in adata.uns:
        rg = adata.uns["rank_genes_groups_cov"]
        t50 = adata.uns["top50_DEGs"]
        deg_ok = True
        for ok, genes in t50.items():
            nk = ok.replace("|", "_")
            if nk not in rg or list(rg[nk]) != list(genes):
                deg_ok = False
        print(f"  Content consistency: {'PASS' if deg_ok else 'FAIL'}")

    # =========================================================================
    # 6. OOD definition
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: OOD Definition")
    print("=" * 70)
    ood = adata.obs[adata.obs["split"] == "ood"]
    print(f"  OOD target: PW034_Panobinostat")
    print(f"  OOD cells: {len(ood)}")
    print(f"  OOD conditions: {dict(ood['condition'].value_counts())}")
    print(f"  OOD has control: {(ood['neg_control'] == 1).any()}")
    print(f"  OOD type: unseen patient-drug combination")
    print(f"  NOT strict unseen patient: PW034 cells (control, Etoposide) exist in train/valid")
    print(f"  NOT strict unseen drug: Panobinostat exists in train/valid for other patients")
    print(f"  Matched control: PW034_control from train/valid")
    pw034_ctrl_n = (adata.obs["cov_drug_name"] == "PW034_control").sum()
    print(f"  PW034_control cells: {pw034_ctrl_n}")

    # =========================================================================
    # 7. Valid groups
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Valid Groups Summary")
    print("=" * 70)
    deg = adata.uns["rank_genes_groups_cov"]
    valid_groups = []
    for gn in sorted(adata.obs["cov_drug_name"].unique()):
        if "control" in gn.lower():
            continue
        nt = (adata.obs["cov_drug_name"] == gn).sum()
        if nt <= 5:
            continue
        if gn not in deg:
            continue
        if len(deg[gn]) < 2:
            continue
        patient = gn.split("_")[0]
        nc = (adata.obs["cov_drug_name"] == f"{patient}_control").sum()
        if nc < 5:
            continue
        is_ood = (gn == "PW034_Panobinostat")
        valid_groups.append((gn, nt, nc, len(deg[gn]), is_ood))

    print(f"  Total valid groups: {len(valid_groups)}")
    print(f"  OOD valid groups: {sum(1 for _, _, _, _, o in valid_groups if o)}")
    for gn, nt, nc, nd, is_ood in valid_groups:
        tag = " [OOD]" if is_ood else ""
        print(f"    {gn}: treated={nt:,d}, matched_ctrl={nc:,d}, DEG={nd}{tag}")

    # =========================================================================
    # 8. Construct example Y_true, Y_ctrl, DEG
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"STEP 8: Construct Example for group='{args.group}'")
    print("=" * 70)

    group = args.group
    if group not in adata.obs["cov_drug_name"].unique():
        print(f"  ERROR: group '{group}' not found in cov_drug_name")
        sys.exit(1)

    # DEG genes
    deg_genes = get_deg_genes(adata, group)
    print(f"  DEG genes: {len(deg_genes)}")

    # treated cells
    treated_mask = adata.obs["cov_drug_name"] == group
    n_treated = treated_mask.sum()
    print(f"  Treated cells: {n_treated}")
    Y_true = adata[treated_mask].X
    print(f"  Y_true (log1p): shape={Y_true.shape}")

    # matched control
    try:
        ctrl_mask = get_matched_control_mask(adata, group)
        n_ctrl = ctrl_mask.sum()
        print(f"  Matched control cells: {n_ctrl}")
        Y_ctrl = adata[ctrl_mask].X
        print(f"  Y_ctrl (log1p): shape={Y_ctrl.shape}")
    except ValueError as e:
        print(f"  WARNING: {e}")

    # =========================================================================
    # 9. Predictions dict example
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Predictions Dict Format")
    print("=" * 70)
    print(f"""
    predictions = {{
        "{group}": {{
            "Y_true": Y_true,           # {n_treated} cells × 5000 log1p expression
            "Y_pred": model_output,     # model prediction, same gene order
            "Y_ctrl": Y_ctrl,           # matched control cells × 5000
            "prediction_space": "{args.prediction_space}",
            "var_names": adata.var_names.tolist(),
            "metadata": {{"method": "your_method_name"}},
        }}
    }}
    """)

    # =========================================================================
    # 10. Key reminders
    # =========================================================================
    print("=" * 70)
    print("KEY REMINDERS")
    print("=" * 70)
    print("""
  1. ONLY use GBM_NIPS_Ready.h5ad (no original/compatible split)
  2. OOD is unseen patient-drug combination (PW034 x Panobinostat)
  3. DO NOT recreate split; use adata.obs['split'] as-is
  4. neg_control == 1 = control, neg_control == 0 = treated
  5. OOD split has no control; matched control is PW034_control from train/valid
  6. rank_genes_groups_cov and top50_DEGs are both available
  7. X_scGPT/XscGPT and X_MolFormer/XMolFormer are both available
  8. Predicted gene order MUST match adata.var_names exactly
  9. Use log1p space for unified evaluation
  10. Counts-space predictions must be converted: Y_pred_eval = log1p(max(Y, 0))
    """)

    print("Done.")


if __name__ == "__main__":
    main()
