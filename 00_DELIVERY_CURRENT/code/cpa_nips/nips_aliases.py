"""NIPS-field aliases shared by CPA runners and evaluators.

The project keeps GBM legacy columns for backward compatibility, but CPA
experiments should read the NIPS-style field names first:
condition, dose_val, cell_type, neg_control, cov_drug_name, rank_genes_groups_cov.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NipsKeys:
    perturbation: str = "condition"
    dosage: str = "dose_val"
    covariate: str = "cell_type"
    control: str = "neg_control"
    group: str = "cov_drug_name"
    degs: str = "rank_genes_groups_cov"
    counts_layer: str = "counts"


KEYS = NipsKeys()


def ensure_nips_aliases(adata, *, add_legacy_aliases: bool = True, require_degs: bool = True):
    """Add missing NIPS-compatible aliases in memory and return key metadata."""
    obs = adata.obs

    if "condition" not in obs and "perturbation" in obs:
        obs["condition"] = obs["perturbation"].astype(str)
    if "dose_val" not in obs and "dosage" in obs:
        obs["dose_val"] = pd.to_numeric(obs["dosage"], errors="coerce").fillna(0.0).astype(float)
    if "cell_type" not in obs and "covariate_patient" in obs:
        obs["cell_type"] = obs["covariate_patient"].astype(str)
    if "neg_control" not in obs and "is_control" in obs:
        obs["neg_control"] = obs["is_control"].astype(int)

    missing = [k for k in ["condition", "dose_val", "cell_type", "neg_control"] if k not in obs]
    if missing:
        raise ValueError(f"Missing NIPS required obs columns and no alias source found: {missing}")

    obs["condition"] = obs["condition"].astype(str)
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["dose_val"] = pd.to_numeric(obs["dose_val"], errors="coerce").fillna(0.0).astype(float)
    obs["neg_control"] = pd.to_numeric(obs["neg_control"], errors="coerce").fillna(0).astype(int)

    if "cov_drug_name" not in obs:
        obs["cov_drug_name"] = obs["cell_type"].astype(str) + "_" + obs["condition"].astype(str)

    if add_legacy_aliases:
        if "perturbation" not in obs:
            obs["perturbation"] = obs["condition"].astype(str)
        if "dosage" not in obs:
            obs["dosage"] = obs["dose_val"].astype(float).astype(str)
        if "covariate_patient" not in obs:
            obs["covariate_patient"] = obs["cell_type"].astype(str)
        if "is_control" not in obs:
            obs["is_control"] = obs["neg_control"].astype(int).eq(1)

    if "rank_genes_groups_cov" not in adata.uns and "top50_DEGs" in adata.uns:
        adata.uns["rank_genes_groups_cov"] = {
            str(k).replace("|", "_", 1): list(v) for k, v in adata.uns["top50_DEGs"].items()
        }
    if add_legacy_aliases and "top50_DEGs" not in adata.uns and "rank_genes_groups_cov" in adata.uns:
        adata.uns["top50_DEGs"] = {
            str(k).replace("_", "|", 1): list(v) for k, v in adata.uns["rank_genes_groups_cov"].items()
        }

    if require_degs and "rank_genes_groups_cov" not in adata.uns:
        raise ValueError("Missing adata.uns['rank_genes_groups_cov'] and no top50_DEGs alias source found")

    return KEYS


def add_full_nips_obs_aliases(adata):
    """Add original-NIPS-shape obs aliases that are meaningful for GBM delivery."""
    ensure_nips_aliases(adata, add_legacy_aliases=True)
    obs = adata.obs

    if "control" not in obs:
        obs["control"] = obs["neg_control"].astype(int).eq(1)
    if "sm_name" not in obs:
        obs["sm_name"] = obs["condition"].astype(str)
    if "sm_lincs_id" not in obs:
        obs["sm_lincs_id"] = obs["condition"].astype(str)
    if "donor_id" not in obs:
        obs["donor_id"] = obs.get("covariate_patient", obs["cell_type"]).astype(str)
    if "type_donor" not in obs:
        obs["type_donor"] = obs["cell_type"].astype(str) + "_" + obs["donor_id"].astype(str)
    if "dose_uM" not in obs:
        obs["dose_uM"] = obs["dose_val"].astype(float)
    if "timepoint_hr" not in obs:
        obs["timepoint_hr"] = 0.0
    if "cov_drug_dose_name" not in obs:
        obs["cov_drug_dose_name"] = (
            obs["cov_drug_name"].astype(str) + "_" + obs["dose_val"].astype(float).astype(str)
        )
    if "obs_id" not in obs:
        obs["obs_id"] = adata.obs_names.astype(str)
    if "cell_id" not in obs:
        obs["cell_id"] = obs["barcode_original"].astype(str) if "barcode_original" in obs else adata.obs_names.astype(str)
    if "library_id" not in obs:
        source = "dataset" if "dataset" in obs else "gsm_accession" if "gsm_accession" in obs else None
        obs["library_id"] = obs[source].astype(str) if source else "GBM"
    if "plate_name" not in obs:
        obs["plate_name"] = obs["library_id"].astype(str)
    if "split2" not in obs:
        obs["split2"] = obs["split"].astype(str)
    if "split3" not in obs:
        obs["split3"] = obs["split"].astype(str)

    for col in ["well", "row", "col", "CLid", "Target", "Pathway", "Disease", "Pathway_3", "Pathway_2"]:
        if col not in obs:
            obs[col] = "NA"

    adata.uns["nips_alias_metadata"] = {
        "schema": "nips_original_field_aliases",
        "note": (
            "GBM-specific aliases for NIPS-compatible code. split2/split3 mirror split; "
            "placeholder metadata columns are set to 'NA' where GBM has no original NIPS equivalent."
        ),
    }
    return adata


def control_mask(obs) -> np.ndarray:
    if "neg_control" in obs:
        return pd.to_numeric(obs["neg_control"], errors="coerce").fillna(0).astype(int).eq(1).to_numpy()
    if "is_control" in obs:
        return obs["is_control"].astype(bool).to_numpy()
    raise KeyError("Need obs['neg_control'] or obs['is_control'] to identify controls")


def group_name(cell_type: str, condition: str) -> str:
    return f"{cell_type}_{condition}"


def deg_genes(adata, cell_type: str, condition: str):
    group = group_name(cell_type, condition)
    if "rank_genes_groups_cov" in adata.uns and group in adata.uns["rank_genes_groups_cov"]:
        return list(adata.uns["rank_genes_groups_cov"][group])
    legacy_key = f"{cell_type}|{condition}"
    if "top50_DEGs" in adata.uns and legacy_key in adata.uns["top50_DEGs"]:
        return list(adata.uns["top50_DEGs"][legacy_key])
    raise KeyError(f"Missing DEG entry for {group} / {legacy_key}")


def set_counterfactual_obs(obs, *, cell_type: str, condition: str, dose: float):
    obs["condition"] = condition
    obs["dose_val"] = float(dose)
    obs["neg_control"] = 0
    obs["cov_drug_name"] = group_name(cell_type, condition)
    obs["cov_drug_dose_name"] = obs["cov_drug_name"].astype(str) + "_" + str(float(dose))
    obs["perturbation"] = condition
    obs["dosage"] = str(float(dose))
    obs["covariate_patient"] = cell_type
    obs["is_control"] = False
    return obs
