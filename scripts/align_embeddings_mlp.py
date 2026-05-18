#!/usr/bin/env python
"""MLP 维度对齐工具。

当 scGPT (512d) 或 MolFormer (768d) embedding 维度与下游模型输入维度不匹配时，
使用 MLP 进行维度投影对齐。

核心功能：
  - align_to_dim(embeddings, target_dim): 用单层或多层 MLP 将对齐到目标维度
  - Aligner 类：可训练的 MLP 对齐模块，支持保存/加载

使用场景示例：
  - scGPT 512d → target_dim=5000 (对齐到原始基因表达空间)
  - MolFormer 768d → target_dim=201 (对齐到 RDKit 指纹空间)
  - MolFormer 768d → target_dim=5000 (对齐到基因表达空间)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class Aligner(nn.Module):
    """MLP 维度对齐模块。

    Parameters
    ----------
    input_dim : int
        输入 embedding 维度（如 512 for scGPT, 768 for MolFormer）
    output_dim : int
        目标维度（如 5000 for gene expression space）
    hidden_dims : Sequence[int], optional
        隐藏层维度列表。默认 [1024, 2048] 逐层渐进。
    dropout : float
        Dropout rate，默认 0.0
    activation : str
        激活函数，默认 'relu'
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (1024, 2048),
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def align_to_dim(
    embeddings: np.ndarray,
    target_dim: int,
    hidden_dims: Sequence[int] = (1024, 2048),
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """便捷函数：用 MLP 投影到目标维度。

    注意：返回未经训练的随机投影。如需训练，请使用 Aligner 类并自行训练。

    Parameters
    ----------
    embeddings : np.ndarray
        输入 embedding，shape (n_samples, input_dim)
    target_dim : int
        目标维度
    hidden_dims : Sequence[int]
        隐藏层维度
    device : str
        计算设备
    seed : int
        随机种子

    Returns
    -------
    np.ndarray, shape (n_samples, target_dim)
    """
    torch.manual_seed(seed)
    input_dim = embeddings.shape[1]
    aligner = Aligner(input_dim, target_dim, hidden_dims=hidden_dims)
    aligner.to(device)

    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        result = aligner(x).cpu().numpy()
    return result


def train_aligner(
    aligner: Aligner,
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> Aligner:
    """训练对齐器以最小化投影嵌入与目标嵌入之间的 MSE。

    Parameters
    ----------
    aligner : Aligner
        待训练的对齐模块
    source_embeddings : np.ndarray
        源嵌入 (n_samples, input_dim)
    target_embeddings : np.ndarray
        目标嵌入 (n_samples, output_dim)
    epochs : int
        训练轮数
    lr : float
        学习率
    batch_size : int
        批次大小
    device : str
        计算设备
    verbose : bool
        是否打印训练信息

    Returns
    -------
    Aligner (已训练)
    """
    aligner.train()
    aligner.to(device)

    x = torch.as_tensor(source_embeddings, dtype=torch.float32, device=device)
    y = torch.as_tensor(target_embeddings, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(aligner.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_samples = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            x_batch, y_batch = x[idx], y[idx]

            optimizer.zero_grad()
            pred = aligner(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch + 1}/{epochs}, loss: {epoch_loss / n_batches:.6f}")

    aligner.eval()
    return aligner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP dimension alignment for embeddings")
    sub = parser.add_subparsers(dest="command")

    # align command
    align_parser = sub.add_parser("align", help="Project embeddings to target dimension (random init)")
    align_parser.add_argument("--input", type=Path, required=True, help="Input .npy file")
    align_parser.add_argument("--output", type=Path, required=True, help="Output .npy file")
    align_parser.add_argument("--target-dim", type=int, required=True, help="Target dimension")
    align_parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 2048])
    align_parser.add_argument("--seed", type=int, default=42)
    align_parser.add_argument("--device", default="cpu")

    # info command
    info_parser = sub.add_parser("info", help="Print info about an embedding file")
    info_parser.add_argument("--input", type=Path, required=True, help="Input .npy file")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "align":
        print(f"Loading: {args.input}")
        embeddings = np.load(args.input)
        print(f"  shape: {embeddings.shape}")

        result = align_to_dim(
            embeddings,
            target_dim=args.target_dim,
            hidden_dims=tuple(args.hidden_dims),
            device=args.device,
            seed=args.seed,
        )
        print(f"  output shape: {result.shape}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)
        print(f"Saved: {output_path}")

    elif args.command == "info":
        data = np.load(args.input)
        print(f"File: {args.input}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Min: {data.min():.6f}, Max: {data.max():.6f}, Mean: {data.mean():.6f}")
        print(f"Non-zero ratio: {(data != 0).mean():.4f}")

    else:
        print("Usage: python align_embeddings_mlp.py {align, info} [args]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
