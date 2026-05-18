#!/usr/bin/env python
"""用 MolFormer 对 GBM 数据集药物 SMILES 编码，生成 768d embedding。

运行环境：conda activate nature

读取原始 GBM h5ad（通过 h5py 绕过 anndata 版本限制）获取药物 SMILES，
用 ibm/MoLFormer-XL-both-10pct 编码后保存：
  - X_MolFormer per-cell embeddings → GBM_X_MolFormer.npy
  - drug-level parquet → GBM_molformer_drug_emb.parquet

注意：此脚本只产出中间文件，最终合并由 plknature 环境执行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode GBM drug SMILES with MolFormer"
    )
    parser.add_argument(
        "--adata",
        type=Path,
        default=ROOT / "GBM_Universal_Perturbation_Ready.h5ad",
        help="Input GBM h5ad (read via h5py for compatibility).",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=ROOT / "GBM_X_MolFormer.npy",
        help="Output per-cell MolFormer embeddings (npy).",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=ROOT / "GBM_molformer_drug_emb.parquet",
        help="Output drug-level parquet (SMILES-indexed, 768 columns).",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=ROOT / "GBM_molformer_drug_emb.metadata.json",
        help="JSON file mapping drug name -> SMILES -> embedding index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for MolFormer inference.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> str:
    return Chem.CanonSmiles(smiles)


def read_drug_smiles_mapping(h5ad_path: Path) -> dict[str, str]:
    """Read drug_name -> SMILES from uns/drug_smiles via h5py."""
    with h5py.File(h5ad_path, "r") as f:
        group = f["uns/drug_smiles"]
        result = {}
        for drug_name in group.keys():
            val = group[drug_name][()]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            result[drug_name] = val
    return result


def read_perturbation_and_control(h5ad_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read perturbation codes, categories, and is_control from obs via h5py.

    Returns:
        pert_codes: (n_cells,) int array
        pert_categories: (n_categories,) str array
        is_control: (n_cells,) bool array
    """
    with h5py.File(h5ad_path, "r") as f:
        pert_group = f["obs/perturbation"]
        codes = pert_group["codes"][()].astype(np.int64)
        categories = pert_group["categories"][()]
        categories = np.array([c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in categories])
        is_control = f["obs/is_control"][()].astype(bool)
    return codes, categories, is_control


def build_per_cell_smiles(
    pert_codes: np.ndarray,
    pert_categories: np.ndarray,
    drug_smiles: dict[str, str],
    is_control: np.ndarray,
) -> np.ndarray:
    """Build per-cell SMILES array from perturbation codes and drug_smiles mapping."""
    # Build category_name -> SMILES lookup
    cat_to_smiles: dict[str, str] = {}
    for cat_name in pert_categories:
        cat_to_smiles[cat_name] = drug_smiles.get(cat_name, "")
    # For control, set empty SMILES
    cat_to_smiles["control"] = ""

    smiles_arr = np.array([cat_to_smiles.get(cat, "") for cat in pert_categories[pert_codes]])
    return smiles_arr


def main() -> None:
    args = parse_args()

    adata_path = Path(args.adata).resolve()
    if not adata_path.exists():
        raise FileNotFoundError(f"Input h5ad not found: {adata_path}")

    print(f"Reading drug SMILES from: {adata_path}")
    drug_smiles = read_drug_smiles_mapping(adata_path)
    print(f"  {len(drug_smiles)} drugs:")
    for name, smi in drug_smiles.items():
        print(f"    {name}: {smi}")

    pert_codes, pert_categories, is_control = read_perturbation_and_control(adata_path)
    print(f"  {len(pert_codes)} cells")
    print(f"  perturbations: {pert_categories.tolist()}")

    per_cell_smiles = build_per_cell_smiles(pert_codes, pert_categories, drug_smiles, is_control)
    print(f"  per-cell SMILES built, {len(per_cell_smiles)} entries")

    unique_smiles = sorted(set(per_cell_smiles) - {""})
    print(f"  unique non-control SMILES: {len(unique_smiles)}")

    # --- MolFormer encoding ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading MolFormer on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    model.to(device)
    model.eval()

    smiles_to_emb: dict[str, np.ndarray] = {}
    print(f"Encoding {len(unique_smiles)} unique SMILES ...")
    for i in tqdm(range(0, len(unique_smiles), args.batch_size), desc="MolFormer"):
        batch = unique_smiles[i : i + args.batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.pooler_output.cpu().numpy()
        for smi, emb in zip(batch, embs):
            smiles_to_emb[smi] = emb

    embed_dim = next(iter(smiles_to_emb.values())).shape[0]
    print(f"  embedding dimension: {embed_dim}")

    # --- Per-cell embeddings ---
    zero_emb = np.zeros(embed_dim, dtype=np.float32)
    cell_embs = np.array(
        [smiles_to_emb.get(smi, zero_emb) for smi in per_cell_smiles],
        dtype=np.float32,
    )
    print(f"  per-cell embedding shape: {cell_embs.shape}")

    output_npy = Path(args.output_npy).resolve()
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, cell_embs)
    print(f"Saved per-cell embeddings: {output_npy}")

    # --- Drug-level parquet ---
    canonical_smiles = sorted({canonicalize_smiles(smi) for smi in unique_smiles})
    drug_emb_rows = []
    for smi in canonical_smiles:
        emb = smiles_to_emb.get(smi)
        if emb is None:
            # Try to find the original SMILE that canonicalizes to this
            for orig_smi, e in smiles_to_emb.items():
                if canonicalize_smiles(orig_smi) == smi:
                    emb = e
                    break
        if emb is not None:
            drug_emb_rows.append((smi, emb))

    col_names = list(range(embed_dim))
    index_smiles = [smi for smi, _ in drug_emb_rows]
    data = np.stack([emb for _, emb in drug_emb_rows], axis=0)
    df = pd.DataFrame(data, index=pd.Index(index_smiles, name="SMILES"), columns=col_names)

    output_parquet = Path(args.output_parquet).resolve()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet)
    print(f"Saved drug-level parquet: {output_parquet}")
    print(f"  shape: {df.shape}")

    # --- Metadata JSON ---
    drug_meta: dict[str, dict] = {}
    for drug_name, smi in drug_smiles.items():
        canonical = canonicalize_smiles(smi)
        drug_meta[drug_name] = {
            "original_smiles": smi,
            "canonical_smiles": canonical,
            "in_parquet": canonical in df.index,
        }

    output_meta = Path(args.output_meta).resolve()
    with open(output_meta, "w") as f:
        json.dump(drug_meta, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata: {output_meta}")


if __name__ == "__main__":
    main()
