#!/usr/bin/env python
"""Train MLP aligner: scGPT embedding (512d) → gene expression (5000d).

Memory-efficient version: only loads expression data for train/valid subsets.

Usage:
  conda activate plknature
  CUDA_VISIBLE_DEVICES=0 python 05_CODE/comparison/train_scgpt_aligner.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse, stats

ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "00_DELIVERY_CURRENT"
REUSABLE = ROOT / "01_REUSABLE_ASSETS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=DELIVERY / "dataset" / "GBM_NIPS_Ready.h5ad")
    parser.add_argument("--aligned-output", type=Path, default=REUSABLE / "embeddings" / "GBM_scGPT_aligned_XscGPT.npy")
    parser.add_argument("--model-out", type=Path, default=REUSABLE / "embeddings" / "GBM_scGPT_aligner.pt")
    parser.add_argument("--scgpt-key", default="X_scGPT",
                        choices=["X_scGPT", "X_scGPT_ctrl", "X_scGPT_pert"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


class Aligner(nn.Module):
    def __init__(self, input_dim=512, output_dim=5000, hidden_dims=(1024, 2048, 4096), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log = open(args.aligned_output.with_suffix(".log"), "w")

    def log_print(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    log_print(f"Device: {device}")

    # Load data in backed mode
    adata = ad.read_h5ad(args.adata, backed="r")
    n_genes = adata.n_vars
    n_cells = adata.n_obs

    train_mask = (adata.obs["split"] == "train").to_numpy()
    valid_mask = (adata.obs["split"] == "valid").to_numpy()
    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)

    log_print(f"Train cells: {len(train_idx)}, Valid cells: {len(valid_idx)}")

    # Load scGPT embeddings (all cells, this is in obsm and loads fast)
    X_scgpt_all = adata.obsm[args.scgpt_key].astype(np.float32)
    scgpt_dim = X_scgpt_all.shape[1]
    log_print(f"scGPT dim: {scgpt_dim}, Key: {args.scgpt_key}")

    # Load gene expression ONLY for train + valid
    train_valid_idx = np.sort(np.concatenate([train_idx, valid_idx]))
    log_print(f"Loading expression for {len(train_valid_idx)} cells...")
    X_expr_subset = dense(adata.X[train_valid_idx]).astype(np.float32)
    log_print(f"Expression loaded: {X_expr_subset.shape}")

    # Create index mapping
    idx_to_pos = {orig: pos for pos, orig in enumerate(train_valid_idx)}
    train_pos = np.array([idx_to_pos[i] for i in train_idx])
    valid_pos = np.array([idx_to_pos[i] for i in valid_idx])

    X_train = X_scgpt_all[train_idx]
    Y_train = X_expr_subset[train_pos]
    X_valid = X_scgpt_all[valid_idx]
    Y_valid = X_expr_subset[valid_pos]

    del X_expr_subset  # Free memory
    log_print(f"Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}")

    # Build aligner
    model = Aligner(input_dim=scgpt_dim, output_dim=n_genes)
    model.to(device)
    log_print(f"Aligner params: {sum(p.numel() for p in model.parameters()):,}")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32).to(device)
    Y_valid_t = torch.tensor(Y_valid, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)
    criterion = nn.MSELoss()

    n_train = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    patience = 40
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = X_train_t[idx].to(device)
            yb = Y_train_t[idx].to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_valid_t)
            val_loss = criterion(val_pred, Y_valid_t).item()
            vp = val_pred.cpu().numpy()
            vy = Y_valid_t.cpu().numpy()
            val_r = stats.pearsonr(vp.mean(axis=0), vy.mean(axis=0)).statistic

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log_print(f"Early stop at epoch {epoch+1}, best_val_loss={best_val_loss:.6f}, val_r={val_r:.4f}")
            break

        if (epoch + 1) % 30 == 0:
            log_print(f"  Epoch {epoch+1}: train={total_loss/n_batches:.6f}, val={val_loss:.6f}, r={val_r:.4f}")

    model.load_state_dict(best_state)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "input_dim": scgpt_dim,
                "output_dim": n_genes, "scgpt_key": args.scgpt_key}, args.model_out)
    log_print(f"Saved aligner: {args.model_out}")

    # Project all cells (batch through to save GPU memory)
    model.eval()
    all_aligned = np.zeros((n_cells, n_genes), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_cells, args.batch_size):
            end = min(start + args.batch_size, n_cells)
            batch = torch.tensor(X_scgpt_all[start:end], dtype=torch.float32).to(device)
            all_aligned[start:end] = model(batch).cpu().numpy()

    all_aligned = np.clip(all_aligned, 0, None)
    np.save(args.aligned_output, all_aligned)
    log_print(f"Saved aligned: {args.aligned_output}, shape={all_aligned.shape}")
    log_print("Done.")
    log.close()


if __name__ == "__main__":
    main()
