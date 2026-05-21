#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
REUSABLE = ROOT / "01_REUSABLE_ASSETS"
RUNTIME = ROOT / "02_RUNTIME_RESULTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=REUSABLE / "preprocessed_data" / "GBM_Universal_Perturbation_Ready.h5ad")
    parser.add_argument("--output", type=Path, default=RUNTIME / "predictions" / "legacy_comparison" / "GBM_MeanShiftBaseline_PW034_Panobinostat_pred.h5ad")
    parser.add_argument("--target-patient", default=None)
    parser.add_argument("--target-drug", default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.adata)
    target = adata.uns.get("ood_split", {})
    patient = args.target_patient or target.get("target_patient")
    drug = args.target_drug or target.get("target_drug")
    if not patient or not drug:
        raise ValueError("Target patient/drug must be provided or present in adata.uns['ood_split']")

    obs = adata.obs
    ood_mask = (obs["cell_type"].eq(patient) & obs["perturbation"].eq(drug)).to_numpy()
    target_ctrl_mask = (obs["cell_type"].eq(patient) & obs["is_control"]).to_numpy()
    source_drug_mask = (~obs["cell_type"].eq(patient) & obs["perturbation"].eq(drug)).to_numpy()
    source_ctrl_mask = (~obs["cell_type"].eq(patient) & obs["is_control"]).to_numpy()
    if not (ood_mask.any() and target_ctrl_mask.any() and source_drug_mask.any() and source_ctrl_mask.any()):
        raise ValueError(f"Cannot build mean-shift baseline for {patient}|{drug}")

    source_delta = dense(adata.X[source_drug_mask]).mean(axis=0) - dense(adata.X[source_ctrl_mask]).mean(axis=0)
    ctrl_idx = np.flatnonzero(target_ctrl_mask)
    rng = np.random.default_rng(args.seed)
    sampled_ctrl = rng.choice(ctrl_idx, size=int(ood_mask.sum()), replace=ctrl_idx.size < int(ood_mask.sum()))
    pred_x = dense(adata.X[sampled_ctrl]) + source_delta
    pred_x = np.clip(pred_x, 0.0, None).astype(np.float32)
    pred = ad.AnnData(X=pred_x, obs=obs.loc[ood_mask].copy(), var=adata.var.copy())
    pred.obs["prediction_method"] = "MeanShiftBaseline"
    pred.uns["prediction_note"] = (
        "PW034 control cells plus the average Panobinostat-control log-expression shift estimated from non-PW034 cells."
    )
    pred.write_h5ad(args.output, compression="gzip")
    print(args.output)


if __name__ == "__main__":
    main()
