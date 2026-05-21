#!/usr/bin/env python
"""Add original-NIPS-style obs aliases to a GBM NIPS-ready h5ad.

This keeps the expression matrices and embeddings unchanged. It only adds
metadata aliases such as dose_val, donor_id, sm_name, type_donor, split2/split3,
and preserves legacy GBM/CPA columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

try:
    from nips_aliases import add_full_nips_obs_aliases
except ModuleNotFoundError:
    from .nips_aliases import add_full_nips_obs_aliases


ROOT = Path(__file__).resolve().parents[3]
DELIVERY = ROOT / "00_DELIVERY_CURRENT"


NIPS_OBS_COLUMNS = [
    "obs_id",
    "library_id",
    "plate_name",
    "well",
    "row",
    "col",
    "cell_id",
    "donor_id",
    "cell_type",
    "sm_lincs_id",
    "sm_name",
    "SMILES",
    "dose_uM",
    "timepoint_hr",
    "control",
    "condition",
    "dose_val",
    "cov_drug_dose_name",
    "cov_drug_name",
    "split",
    "neg_control",
    "type_donor",
    "CLid",
    "split2",
    "split3",
    "Target",
    "Pathway",
    "Disease",
    "Pathway_3",
    "Pathway_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch GBM h5ad with original NIPS field aliases")
    parser.add_argument("--input", type=Path, default=DELIVERY / "dataset" / "GBM_NIPS_Ready.h5ad")
    parser.add_argument("--output", type=Path, default=DELIVERY / "dataset" / "GBM_NIPS_Ready_patched.h5ad")
    parser.add_argument("--compression", default="gzip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.input)
    adata.obs = adata.obs.copy()
    add_full_nips_obs_aliases(adata)

    missing = [col for col in NIPS_OBS_COLUMNS if col not in adata.obs.columns]
    if missing:
        raise RuntimeError(f"Failed to create NIPS obs aliases: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression=args.compression)
    print(f"Wrote: {args.output}")
    print(f"Shape: {adata.shape}")
    print(f"NIPS obs aliases: {len(NIPS_OBS_COLUMNS)} / {len(NIPS_OBS_COLUMNS)} present")


if __name__ == "__main__":
    main()
