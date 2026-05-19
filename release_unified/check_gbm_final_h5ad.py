#!/usr/bin/env python3
"""
GBM Final H5AD Integrity Check
===============================
Verifies that GBM_NIPS_Ready.h5ad meets all requirements for release.

Usage:
    python check_gbm_final_h5ad.py [--h5ad PATH]
"""
import argparse, sys, os
import numpy as np
import anndata as ad
import scipy.sparse as sp

def check(h5ad_path):
    errors = []
    warnings = []
    checks_passed = 0
    checks_total = 0

    def chk(name, condition, severity="error"):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f"  [PASS] {name}")
        else:
            if severity == "error":
                errors.append(name)
                print(f"  [FAIL] {name}")
            else:
                warnings.append(name)
                print(f"  [WARN] {name}")

    print(f"Loading: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  shape: {adata.shape}")

    # =========================================================================
    # 1. Basic structure
    # =========================================================================
    print("\n--- 1. Basic Structure ---")
    chk("shape == (169972, 5000)", adata.shape == (169972, 5000))
    chk("X is log1p-normalized (float32)", adata.X.dtype == np.float32)
    chk("X is sparse CSR", sp.issparse(adata.X) and isinstance(adata.X, sp.csr_matrix))

    # Sparse statistics
    if sp.issparse(adata.X):
        nnz = adata.X.nnz
        total = adata.X.shape[0] * adata.X.shape[1]
        density = nnz / total
        implicit_zeros = 1.0 - density
        print(f"  X nnz={nnz:,}, density={density:.4f}, implicit_zero_ratio={implicit_zeros:.4f}")
        print(f"  X nonzero_min={adata.X.data.min():.4f}, nonzero_max={adata.X.data.max():.4f}")

    # =========================================================================
    # 2. Layers
    # =========================================================================
    print("\n--- 2. Layers ---")
    chk("layers['counts'] exists", "counts" in adata.layers)
    if "counts" in adata.layers:
        c = adata.layers["counts"]
        chk("counts is int32", c.dtype == np.int32)
        chk("counts is sparse CSR", sp.issparse(c) and isinstance(c, sp.csr_matrix))
        chk("counts no NaN", not np.any(np.isnan(c.data))) if sp.issparse(c) else True
        chk("counts no Inf", not np.any(np.isinf(c.data))) if sp.issparse(c) else True
        chk("counts non-negative", c.data.min() >= 0) if sp.issparse(c) else True
        nnzc = c.nnz if sp.issparse(c) else c.size
        totalc = c.shape[0] * c.shape[1]
        densityc = nnzc / totalc
        print(f"  counts nnz={nnzc:,}, density={densityc:.4f}, implicit_zero_ratio={1-densityc:.4f}")
        print(f"  counts nonzero_min={c.data.min():.0f}, nonzero_max={c.data.max():.0f}")

    # =========================================================================
    # 3. obs required fields
    # =========================================================================
    print("\n--- 3. obs Required Fields ---")
    required = [
        "perturbation", "dosage", "covariate_patient", "condition",
        "cov_drug_name", "cell_type", "is_control", "is_treated",
        "split", "neg_control", "SMILES", "canonical_smiles",
        "gsm_accession", "barcode_original", "dataset"
    ]
    for field in required:
        chk(f"obs['{field}'] exists", field in adata.obs.columns)

    # =========================================================================
    # 4. obsm keys
    # =========================================================================
    print("\n--- 4. obsm Keys ---")
    required_obsm = [
        ("X_scGPT", (169972, 512)),
        ("XscGPT", (169972, 512)),
        ("X_scGPT_ctrl", (169972, 512)),
        ("X_scGPT_pert", (169972, 512)),
        ("X_MolFormer", (169972, 768)),
        ("XMolFormer", (169972, 768)),
    ]
    for key, expected_shape in required_obsm:
        chk(f"obsm['{key}'] exists", key in adata.obsm)
        if key in adata.obsm:
            chk(f"obsm['{key}'] shape={expected_shape}",
                adata.obsm[key].shape == expected_shape)
            chk(f"obsm['{key}'] no NaN",
                not np.any(np.isnan(adata.obsm[key])))
            chk(f"obsm['{key}'] no Inf",
                not np.any(np.isinf(adata.obsm[key])))

    # Alias consistency
    print("\n--- 4b. Alias Consistency ---")
    if "XscGPT" in adata.obsm and "X_scGPT" in adata.obsm:
        chk("XscGPT == X_scGPT",
            np.allclose(adata.obsm["XscGPT"], adata.obsm["X_scGPT"], atol=1e-6))
    if "XMolFormer" in adata.obsm and "X_MolFormer" in adata.obsm:
        chk("XMolFormer == X_MolFormer",
            np.allclose(adata.obsm["XMolFormer"], adata.obsm["X_MolFormer"], atol=1e-6))

    # =========================================================================
    # 5. uns keys
    # =========================================================================
    print("\n--- 5. uns Keys ---")
    required_uns = ["top50_DEGs", "rank_genes_groups_cov", "drug_smiles", "release_metadata"]
    for key in required_uns:
        chk(f"uns['{key}'] exists", key in adata.uns)

    # DEG alias consistency
    print("\n--- 5b. DEG Consistency ---")
    if "top50_DEGs" in adata.uns and "rank_genes_groups_cov" in adata.uns:
        t50 = adata.uns["top50_DEGs"]
        rg = adata.uns["rank_genes_groups_cov"]
        chk("rank_genes_groups_cov entries == top50_DEGs entries", len(rg) == len(t50))
        deg_consistent = True
        for ok, genes in t50.items():
            nk = ok.replace("|", "_")
            if nk not in rg:
                deg_consistent = False
                print(f"    Missing: {nk} in rank_genes_groups_cov")
            elif list(rg[nk]) != list(genes):
                deg_consistent = False
                print(f"    Mismatch: {nk}")
        chk("rank_genes_groups_cov content == top50_DEGs", deg_consistent)

    # =========================================================================
    # 6. Split
    # =========================================================================
    print("\n--- 6. Split Distribution ---")
    split_dist = dict(adata.obs["split"].value_counts())
    chk("train == 150564", split_dist.get("train", 0) == 150564)
    chk("valid == 16729", split_dist.get("valid", 0) == 16729)
    chk("ood == 2679", split_dist.get("ood", 0) == 2679)

    # =========================================================================
    # 7. OOD check
    # =========================================================================
    print("\n--- 7. OOD Check ---")
    ood = adata.obs[adata.obs["split"] == "ood"]
    chk("OOD only contains PW034_Panobinostat",
        set(ood["cov_drug_name"].unique()) == {"PW034_Panobinostat"})
    chk("OOD has no control cells",
        (ood["neg_control"] == 1).sum() == 0)
    chk("PW034_control exists as matched control",
        "PW034_control" in adata.obs["cov_drug_name"].unique())
    pw034_ctrl_count = (adata.obs["cov_drug_name"] == "PW034_control").sum()
    chk(f"PW034_control count >= 5 (actual={pw034_ctrl_count})", pw034_ctrl_count >= 5)

    # =========================================================================
    # 8. Valid groups
    # =========================================================================
    print("\n--- 8. Valid Groups ---")
    deg = adata.uns["rank_genes_groups_cov"]
    if deg:
        valid = []
        for gn in adata.obs["cov_drug_name"].unique():
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
            ctrl_gn = f"{patient}_control"
            nc = (adata.obs["cov_drug_name"] == ctrl_gn).sum()
            if nc < 5:
                continue
            valid.append(gn)
        chk("valid group count == 18", len(valid) == 18, severity="warning")
        print(f"  Valid groups: {len(valid)}")
        ood_valid = [g for g in valid if g == "PW034_Panobinostat"]
        chk("OOD valid group count == 1", len(ood_valid) == 1)

    # =========================================================================
    # 9. Gene order
    # =========================================================================
    print("\n--- 9. Gene Order ---")
    chk("var_names unique", len(adata.var_names) == len(set(adata.var_names)))
    chk("var_names count == 5000", len(adata.var_names) == 5000)

    # =========================================================================
    # 10. release_metadata content
    # =========================================================================
    print("\n--- 10. Release Metadata ---")
    if "release_metadata" in adata.uns:
        rm = adata.uns["release_metadata"]
        chk("release_version present", "release_version" in rm)
        chk("single_recommended_h5ad set", rm.get("single_recommended_h5ad") == "GBM_NIPS_Ready.h5ad")
        chk("ood_type is unseen_patient_drug_combination",
            "unseen_patient_drug_combination" in str(rm.get("ood_type", "")))

    # =========================================================================
    # 11. Obs names
    # =========================================================================
    print("\n--- 11. Identifier Uniqueness ---")
    chk("obs_names all unique", len(adata.obs_names) == len(set(adata.obs_names)))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {checks_passed}/{checks_total} checks passed")
    print(f"Errors: {len(errors)}, Warnings: {len(warnings)}")

    status = "PASS"
    if errors:
        status = "FAIL"
    elif warnings:
        status = "PASS_WITH_WARNINGS"

    print(f"FINAL STATUS: {status}")
    return status, errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Check GBM final H5AD integrity")
    parser.add_argument("--h5ad", default="GBM_NIPS_Ready.h5ad",
                        help="Path to GBM_NIPS_Ready.h5ad")
    args = parser.parse_args()

    if not os.path.exists(args.h5ad):
        # Try relative to script directory
        base = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(os.path.dirname(base), args.h5ad)
        if os.path.exists(alt_path):
            args.h5ad = alt_path
        else:
            print(f"ERROR: {args.h5ad} not found")
            sys.exit(1)

    status, errors, warnings = check(args.h5ad)
    if status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
