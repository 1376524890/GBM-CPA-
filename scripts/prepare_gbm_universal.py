#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "GBM_dataset" / "GSE148842_cpa_ready.h5ad",
    ROOT / "GBM_dataset" / "GSE226202_cpa_ready.h5ad",
]
DEFAULT_OUTPUT = ROOT / "GBM_Universal_Perturbation_Ready.h5ad"

FALLBACK_SMILES = {
    "Ana-12": "CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2C4=CC=CC=C4",
    "Etoposide": "COC1=CC2=C(C=C1O)C3C(COC3=O)C4=CC5=C(C=C4O2)OCO5",
    "Ispenisib": "CC(C)N1CCN(CC1)C2=NC(=NC=C2)N3CCOCC3",
    "Panobinostat": "C1CN(CCC1C(=O)NO)CCN2C=C(C=N2)C3=CC=C(C=C3)C=C",
    "RO4929097": "CC(C)(C)OC(=O)N1CCC(CC1)N2C=NC3=C2C=C(C=C3)C(F)(F)F",
    "Tazemetostat": "CC(C)N1CCN(CC1)C2=NC(=NC=C2)N3CCOCC3",
    "Temozolomide": "CN1C(=O)N2C=NC(=N2)N=N1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-patient", default="PW034")
    parser.add_argument("--target-drug", default="Panobinostat")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-hvg", type=int, default=5000)
    return parser.parse_args()


def as_csr(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def integer_count_layer(adata: ad.AnnData, dataset_id: str) -> sparse.csr_matrix:
    counts = as_csr(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    counts = counts.astype(np.float64)
    rounded = counts.copy()
    rounded.data = np.rint(rounded.data)
    rounded.data[rounded.data < 0] = 0
    if not np.allclose(counts.data, rounded.data, rtol=0, atol=0.51):
        raise ValueError(f"{dataset_id} counts contain values too far from integers to round safely")
    return rounded.astype(np.int32)


def standardize_one(path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    dataset_id = str(adata.uns.get("dataset_id", path.stem.split("_")[0]))
    adata.obs = adata.obs.copy()
    adata.obs["dataset"] = dataset_id
    adata.obs["perturbation"] = adata.obs["perturbation"].astype(str).replace({"DMSO": "control", "dmso": "control"})
    adata.obs.loc[adata.obs["perturbation"].str.lower().isin(["none", "vehicle", "vehicle (dmso)"]), "perturbation"] = "control"
    adata.obs["dosage"] = pd.to_numeric(adata.obs["dosage"], errors="coerce").fillna(0.0).astype(float)
    adata.obs["covariate_patient"] = adata.obs["covariate_patient"].astype(str)
    adata.obs["cell_type"] = adata.obs["covariate_patient"]
    adata.obs["is_control"] = adata.obs["perturbation"].eq("control")
    adata.layers["counts"] = integer_count_layer(adata, dataset_id)
    adata.X = adata.layers["counts"].copy().astype(np.float32)
    return adata


def normalize_and_hvg(adata: ad.AnnData, n_hvg: int) -> ad.AnnData:
    adata.X = adata.layers["counts"].copy().astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_hvg = min(n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    return adata[:, adata.var["highly_variable"].to_numpy()].copy()


def mean_vector(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0)


def var_vector(x) -> np.ndarray:
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
        return np.maximum(mean_sq - mean * mean, 0.0)
    return np.asarray(x).var(axis=0)


def calculate_top50_degs(adata: ad.AnnData) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    genes = adata.var_names.to_numpy()
    for patient in sorted(adata.obs["cell_type"].unique()):
        ctrl_mask = (adata.obs["cell_type"] == patient).to_numpy() & adata.obs["is_control"].to_numpy()
        n_ctrl = int(ctrl_mask.sum())
        if n_ctrl < 2:
            continue
        x_ctrl = adata.X[ctrl_mask]
        ctrl_mean = mean_vector(x_ctrl)
        ctrl_var = var_vector(x_ctrl)
        for drug in sorted(adata.obs.loc[adata.obs["cell_type"].eq(patient), "perturbation"].unique()):
            if drug == "control":
                continue
            pert_mask = (adata.obs["cell_type"] == patient).to_numpy() & (adata.obs["perturbation"] == drug).to_numpy()
            n_pert = int(pert_mask.sum())
            if n_pert < 2:
                continue
            x_pert = adata.X[pert_mask]
            pert_mean = mean_vector(x_pert)
            pert_var = var_vector(x_pert)
            denom = np.sqrt(pert_var / n_pert + ctrl_var / n_ctrl)
            t_stat = np.divide(pert_mean - ctrl_mean, denom, out=np.zeros_like(denom), where=denom > 0)
            pvals = 2.0 * stats.norm.sf(np.abs(t_stat))
            padj = multipletests(pvals, method="fdr_bh")[1]
            logfc = np.log2((pert_mean + 1e-8) / (ctrl_mean + 1e-8))
            order = np.lexsort((padj, -np.abs(logfc)))
            result[f"{patient}|{drug}"] = genes[order[:50]].astype(str).tolist()
    return result


def pubchem_smiles(name: str) -> str | None:
    encoded = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT"
    try:
        with urllib.request.urlopen(url, timeout=12) as handle:
            value = handle.read().decode("utf-8").strip().splitlines()[0]
            return value or None
    except Exception:
        return None


def rdkit_embeddings(drugs: list[str]) -> tuple[dict[str, str], dict[str, list[float]]]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdMolDescriptors

    smiles: dict[str, str] = {}
    embeddings: dict[str, list[float]] = {}
    for drug in drugs:
        smi = pubchem_smiles(drug) or FALLBACK_SMILES.get(drug)
        if not smi:
            raise RuntimeError(f"No SMILES available for {drug}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"RDKit cannot parse SMILES for {drug}: {smi}")
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=196)
        bits = np.zeros((196,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, bits)
        desc = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                float(Descriptors.NumHDonors(mol)),
                float(Descriptors.NumHAcceptors(mol)),
            ],
            dtype=np.float32,
        )
        desc = np.nan_to_num(desc)
        emb = np.concatenate([desc, bits]).astype(np.float32)
        if emb.shape[0] != 201:
            raise AssertionError("Expected 201-dimensional drug embedding")
        smiles[drug] = smi
        embeddings[drug] = emb.tolist()
    return smiles, embeddings


def assign_split(adata: ad.AnnData, target_patient: str, target_drug: str, valid_fraction: float, seed: int) -> None:
    split = np.full(adata.n_obs, "train", dtype=object)
    ood = (adata.obs["cell_type"].eq(target_patient) & adata.obs["perturbation"].eq(target_drug)).to_numpy()
    if not ood.any():
        raise ValueError(f"OOD combination not found: {target_patient}|{target_drug}")
    split[ood] = "ood"
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(~ood)
    n_valid = max(1, math.floor(train_idx.size * valid_fraction))
    valid_idx = rng.choice(train_idx, size=n_valid, replace=False)
    split[valid_idx] = "valid"
    adata.obs["split"] = split


def main() -> None:
    args = parse_args()
    parts = [standardize_one(path) for path in args.inputs]
    adata = ad.concat(parts, join="inner", merge="same", label="source_file", keys=[p.stem for p in args.inputs])
    adata.obs_names_make_unique()
    adata = normalize_and_hvg(adata, args.n_hvg)
    assign_split(adata, args.target_patient, args.target_drug, args.valid_fraction, args.seed)
    adata.uns["top50_DEGs"] = calculate_top50_degs(adata)
    drugs = sorted([x for x in adata.obs["perturbation"].unique() if x != "control"])
    smiles, embeddings = rdkit_embeddings(drugs)
    adata.uns["drug_smiles"] = smiles
    adata.uns["drug_embeddings"] = embeddings
    adata.obs["foundation_model_query_eligible"] = adata.obs["is_control"].astype(bool)
    adata.uns["foundation_model_query"] = {
        "control_mask_obs_key": "foundation_model_query_eligible",
        "expression_matrix": "X_log1p_normalized_1e4",
        "gene_axis": "var_names",
    }
    adata.uns["ood_split"] = {"target_patient": args.target_patient, "target_drug": args.target_drug}
    adata.uns["count_layer_note"] = (
        "layers['counts'] stores unnormalized integer counts. GSE226202 publisher files contain near-integer corrected "
        "values, rounded to nearest integer to satisfy integer-count model interfaces."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression="gzip")
    summary = {
        "output": str(args.output),
        "shape": list(adata.shape),
        "split_counts": adata.obs["split"].value_counts().to_dict(),
        "top50_deg_conditions": len(adata.uns["top50_DEGs"]),
        "drug_embedding_dims": {k: len(v) for k, v in embeddings.items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
