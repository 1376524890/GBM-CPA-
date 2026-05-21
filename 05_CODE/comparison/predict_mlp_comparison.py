#!/usr/bin/env python
"""MLP-based counterfactual perturbation predictors for embedding comparison.

Trains simple MLPs using different cell/drug embedding combinations to predict
perturbation effects, enabling comparison of representation quality.

Methods:
  M1: scGPT cell (512d) + learnable drug embedding
  M2: scGPT ctrl-only (512d) + learnable drug embedding
  M3: scGPT pert-only (512d) + learnable drug embedding
  M5: scGPT cell (512d) + MolFormer drug (768d)

Usage:
  conda activate plknature
  python 05_CODE/comparison/predict_mlp_comparison.py --method M1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse, stats
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
REUSABLE = ROOT / "01_REUSABLE_ASSETS"
RUNTIME = ROOT / "02_RUNTIME_RESULTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=REUSABLE / "preprocessed_data" / "GBM_with_embeddings.h5ad")
    parser.add_argument("--molformer-parquet", type=Path, default=REUSABLE / "embeddings" / "GBM_molformer_drug_emb.parquet")
    parser.add_argument("--output-dir", type=Path, default=RUNTIME / "predictions" / "legacy_comparison" / "mlp_comparison_results")
    parser.add_argument("--method", required=True,
                        choices=["M1", "M2", "M3", "M5"],
                        help="Which embedding configuration to use")
    parser.add_argument("--target-patient", default="PW034")
    parser.add_argument("--target-drug", default="Panobinostat")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sinkhorn-samples", type=int, default=512)
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True))


class PerturbationMLP(nn.Module):
    """MLP that predicts post-perturbation gene expression from
    (cell_embedding, drug_embedding)."""

    def __init__(self, cell_dim: int, drug_dim: int, n_genes: int,
                 hidden_dims=(2048, 4096, 2048), dropout=0.1):
        super().__init__()
        in_dim = cell_dim + drug_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_genes))
        self.net = nn.Sequential(*layers)

    def forward(self, cell_emb, drug_emb):
        x = torch.cat([cell_emb, drug_emb], dim=-1)
        return self.net(x)


def load_embeddings(args):
    """Load the data and extract the appropriate embeddings for the method."""
    adata = ad.read_h5ad(args.adata, backed="r")

    n_genes = adata.n_vars
    obs = adata.obs

    # Determine cell embedding key
    cell_key_map = {
        "M1": "X_scGPT",
        "M2": "X_scGPT_ctrl",
        "M3": "X_scGPT_pert",
        "M5": "X_scGPT",
    }
    cell_key = cell_key_map[args.method]
    cell_emb = adata.obsm[cell_key].copy()
    cell_dim = cell_emb.shape[1]

    # Drug embeddings
    if args.method in ("M1", "M2", "M3"):
        # Learnable drug embeddings will be trained as part of the MLP
        drug_names = sorted(obs["perturbation"].unique())
        drug_to_idx = {d: i for i, d in enumerate(drug_names)}
        drug_dim = len(drug_names)  # Will use nn.Embedding in the model
        drug_emb_type = "learnable"
        drug_idx = np.array([drug_to_idx.get(p, 0) for p in obs["perturbation"]])
    else:  # M5: MolFormer
        molformer_emb = adata.obsm["X_MolFormer"].copy()
        drug_dim = molformer_emb.shape[1]
        drug_emb_type = "MolFormer"
        drug_emb = molformer_emb
        drug_idx = None

    return adata, cell_emb, cell_dim, drug_dim, drug_emb_type, drug_idx, n_genes


def prepare_training_data(adata, args, cell_emb, drug_emb_type, drug_idx):
    """Prepare training data: (cell_emb, drug_emb) -> post-treatment expression."""
    obs = adata.obs

    # Training: treated cells from non-OOD patients
    train_mask = (obs["split"] == "train") & (~obs["is_control"])
    # Validation: treated cells from validation split
    valid_mask = (obs["split"] == "valid") & (~obs["is_control"])

    X_cell_train = cell_emb[train_mask.to_numpy()]
    X_cell_valid = cell_emb[valid_mask.to_numpy()]

    # Target: post-treatment gene expression (log1p normalized)
    y_train = dense(adata.X[train_mask.to_numpy()])
    y_valid = dense(adata.X[valid_mask.to_numpy()])

    if drug_emb_type == "learnable":
        X_drug_train = drug_idx[train_mask.to_numpy()]
        X_drug_valid = drug_idx[valid_mask.to_numpy()]
    else:
        molformer = adata.obsm["X_MolFormer"]
        X_drug_train = molformer[train_mask.to_numpy()]
        X_drug_valid = molformer[valid_mask.to_numpy()]

    emit("data_prepared", train_samples=int(train_mask.sum()), valid_samples=int(valid_mask.sum()),
         cell_dim=X_cell_train.shape[1], drug_dim=X_drug_train.shape[1] if drug_emb_type != "learnable" else "learnable")

    return (X_cell_train, X_drug_train, y_train), (X_cell_valid, X_drug_valid, y_valid)


def train_model(model, drug_embedding_layer, train_data, valid_data, args):
    """Train the MLP predictor."""
    X_cell_train, X_drug_train, y_train = train_data
    X_cell_valid, X_drug_valid, y_valid = valid_data

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    if drug_embedding_layer is not None:
        drug_embedding_layer.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + (list(drug_embedding_layer.parameters()) if drug_embedding_layer else []),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_cell_train_t = torch.tensor(X_cell_train, dtype=torch.float32)
    X_cell_valid_t = torch.tensor(X_cell_valid, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid, dtype=torch.float32)

    if drug_embedding_layer is not None:
        X_drug_train_t = torch.tensor(X_drug_train, dtype=torch.long)
        X_drug_valid_t = torch.tensor(X_drug_valid, dtype=torch.long)
    else:
        X_drug_train_t = torch.tensor(X_drug_train, dtype=torch.float32)
        X_drug_valid_t = torch.tensor(X_drug_valid, dtype=torch.float32)

    n_train = X_cell_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        if drug_embedding_layer:
            drug_embedding_layer.train()

        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            cell_batch = X_cell_train_t[idx].to(device)
            y_batch = y_train_t[idx].to(device)

            if drug_embedding_layer is not None:
                drug_idx_batch = X_drug_train_t[idx].to(device)
                drug_batch = drug_embedding_layer(drug_idx_batch)
            else:
                drug_batch = X_drug_train_t[idx].to(device)

            optimizer.zero_grad()
            pred = model(cell_batch, drug_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        if drug_embedding_layer:
            drug_embedding_layer.eval()

        with torch.no_grad():
            cell_val = X_cell_valid_t.to(device)
            y_val = y_valid_t.to(device)
            if drug_embedding_layer is not None:
                drug_val = drug_embedding_layer(X_drug_valid_t.to(device))
            else:
                drug_val = X_drug_valid_t.to(device)

            val_pred = model(cell_val, drug_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            emit("early_stop", epoch=epoch + 1, best_val_loss=best_val_loss)
            break

        if (epoch + 1) % 20 == 0:
            emit("epoch", epoch=epoch + 1, train_loss=total_loss / n_batches, val_loss=val_loss)

    model.load_state_dict(best_state)
    return model


def predict_ood_mlp(adata, args, model, drug_embedding_layer, cell_emb, drug_emb_type, drug_idx):
    """Generate OOD predictions for PW034 + Panobinostat."""
    obs = adata.obs

    # PW034 control cells
    ood_ctrl_mask = (obs["cell_type"] == args.target_patient) & (obs["is_control"])
    ood_ctrl_idx = np.flatnonzero(ood_ctrl_mask.to_numpy())

    # Ground truth OOD cells (for evaluation only, not used in prediction)
    ood_true_mask = (obs["cell_type"] == args.target_patient) & (obs["perturbation"] == args.target_drug)

    X_cell_ood = cell_emb[ood_ctrl_idx]
    y_ctrl = dense(adata.X[ood_ctrl_idx])

    device = next(model.parameters()).device
    model.eval()

    # Get drug embedding for Panobinostat
    if drug_emb_type == "learnable":
        drug_to_idx = {d: i for i, d in enumerate(sorted(obs["perturbation"].unique()))}
        drug_id = drug_to_idx.get(args.target_drug, 0)
        drug_emb_ood = drug_embedding_layer(torch.tensor([drug_id] * len(ood_ctrl_idx)).to(device))
    else:
        # MolFormer: find the drug embedding for Panobinostat
        pert_mask = (obs["perturbation"] == args.target_drug) & (~obs["is_control"])
        if pert_mask.sum() > 0:
            pert_idx = np.flatnonzero(pert_mask.to_numpy())[0]
            molformer = adata.obsm["X_MolFormer"]
            pan_emb = molformer[pert_idx]
        else:
            pan_emb = np.zeros(adata.obsm["X_MolFormer"].shape[1], dtype=np.float32)
        drug_emb_ood = torch.tensor(np.tile(pan_emb, (len(ood_ctrl_idx), 1)), dtype=torch.float32).to(device)

    with torch.no_grad():
        cell_t = torch.tensor(X_cell_ood, dtype=torch.float32).to(device)
        pred_y = model(cell_t, drug_emb_ood).cpu().numpy()

    # Clip to valid range
    pred_y = np.clip(pred_y, 0, None)

    # Create prediction AnnData
    pred_obs = obs.iloc[ood_ctrl_idx].copy()
    pred_obs["perturbation"] = args.target_drug
    pred_obs["is_control"] = False

    pred = ad.AnnData(
        X=pred_y.astype(np.float32),
        obs=pred_obs,
        var=adata.var.copy(),
        uns={"prediction": {"method": f"MLP-{args.method}", "target_patient": args.target_patient,
                            "target_drug": args.target_drug}}
    )
    pred.obs_names = adata.obs_names[ood_ctrl_idx].copy()

    return pred, y_ctrl, dense(adata.X[ood_true_mask.to_numpy()]), adata


def sinkhorn_distance(x_pred: np.ndarray, x_true: np.ndarray, samples: int, seed: int) -> float:
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


def compute_metrics_mlp(args, pred, adata_full):
    """Compute standard CRISP OOD metrics using the full AnnData for ground truth."""
    key = f"{args.target_patient}|{args.target_drug}"
    genes = list(adata_full.uns["top50_DEGs"][key])
    gene_idx = adata_full.var_names.get_indexer(genes)
    pred_idx = pred.var_names.get_indexer(genes)

    true_mask = (adata_full.obs["cell_type"].eq(args.target_patient)
                 & adata_full.obs["perturbation"].eq(args.target_drug)).to_numpy()
    ctrl_mask = (adata_full.obs["cell_type"].eq(args.target_patient)
                 & adata_full.obs["is_control"]).to_numpy()

    x_true = dense(adata_full.X[true_mask][:, gene_idx])
    x_ctrl = dense(adata_full.X[ctrl_mask][:, gene_idx])
    x_pred = dense(pred.X[:, pred_idx])

    true_post = x_true.mean(axis=0)
    pred_post = x_pred.mean(axis=0)
    ctrl = x_ctrl.mean(axis=0)
    true_logfc = true_post - ctrl
    pred_logfc = pred_post - ctrl

    metrics = {
        "method": f"MLP ({args.method})",
        "patient": args.target_patient,
        "drug": args.target_drug,
        "pearson": float(stats.pearsonr(pred_logfc, true_logfc).statistic),
        "spearman": float(stats.spearmanr(pred_logfc, true_logfc).statistic),
        "r2": float(r2_score(true_post, pred_post)),
        "sinkhorn": sinkhorn_distance(x_pred, x_true, args.sinkhorn_samples, args.seed),
        "direction": float(np.mean(np.sign(pred_logfc) == np.sign(true_logfc)) * 100.0),
    }
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    emit("run_start", method=args.method)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    adata, cell_emb, cell_dim, drug_dim, drug_emb_type, drug_idx, n_genes = load_embeddings(args)
    emit("embeddings_loaded", cell_dim=cell_dim, drug_dim=drug_dim, drug_emb_type=drug_emb_type)

    # Prepare training data
    train_data, valid_data = prepare_training_data(adata, args, cell_emb, drug_emb_type, drug_idx)

    # Build model
    model = PerturbationMLP(cell_dim=cell_dim, drug_dim=drug_dim, n_genes=n_genes)

    if drug_emb_type == "learnable":
        n_drugs = len(adata.obs["perturbation"].unique())
        drug_embedding_layer = nn.Embedding(n_drugs, drug_dim)
        # Initialize with small random values
        nn.init.normal_(drug_embedding_layer.weight, std=0.1)
    else:
        drug_embedding_layer = None
        drug_dim = adata.obsm["X_MolFormer"].shape[1]  # Override with actual MolFormer dim

    emit("model_built", total_params=sum(p.numel() for p in model.parameters()))

    # Train
    model = train_model(model, drug_embedding_layer, train_data, valid_data, args)

    # Generate OOD predictions
    pred, y_ctrl, y_true, adata_full = predict_ood_mlp(
        adata, args, model, drug_embedding_layer, cell_emb, drug_emb_type, drug_idx)

    # Save prediction
    pred_path = args.output_dir / f"MLP_{args.method}_PW034_Panobinostat_pred.h5ad"
    pred.write_h5ad(pred_path, compression="gzip")
    emit("prediction_saved", output=str(pred_path), shape=list(pred.shape))

    # Load full AnnData for evaluation
    adata_full_eval = ad.read_h5ad(REUSABLE / "preprocessed_data" / "GBM_Universal_Perturbation_Ready.h5ad")
    adata_full_eval.obs["cell_type"] = adata_full_eval.obs["cell_type"].astype(str)
    adata_full_eval.obs["perturbation"] = adata_full_eval.obs["perturbation"].astype(str)
    adata_full_eval.obs["is_control"] = adata_full_eval.obs["is_control"].astype(bool)

    # Compute metrics
    metrics = compute_metrics_mlp(args, pred, adata_full_eval)

    # Save metrics
    metrics_path = args.output_dir / f"MLP_{args.method}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    emit("run_complete", metrics=metrics)


if __name__ == "__main__":
    main()
