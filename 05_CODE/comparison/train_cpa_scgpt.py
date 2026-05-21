#!/usr/bin/env python
"""Train CPA with scGPT-aligned cell input (M1).

Uses the pre-trained MLP aligner to project scGPT 512d → gene expression 5000d,
then trains CPA on the aligned continuous values using Gaussian reconstruction loss.

Usage:
  conda activate plknature
  CUDA_VISIBLE_DEVICES=0 python 05_CODE/comparison/train_cpa_scgpt.py --scgpt-key X_scGPT --method M1
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import TextIO

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse, stats
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "00_DELIVERY_CURRENT"
REUSABLE = ROOT / "01_REUSABLE_ASSETS"
RUNTIME = ROOT / "02_RUNTIME_RESULTS"

_ALLOWED_CUDA_VISIBLE_DEVICES = {"0", "1", "2", "3"}
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams
    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=DELIVERY / "dataset" / "GBM_NIPS_Ready.h5ad")
    parser.add_argument("--aligner", type=Path, default=REUSABLE / "embeddings" / "GBM_scGPT_aligner.pt")
    parser.add_argument("--scgpt-key", default="X_scGPT",
                        choices=["X_scGPT", "X_scGPT_ctrl", "X_scGPT_pert"])
    parser.add_argument("--method", default="M1", help="Method label")
    parser.add_argument("--model-dir", type=Path, default=RUNTIME / "models" / "legacy_comparison" / "GBM_CPA_scGPT_model")
    parser.add_argument("--predicted", type=Path, default=RUNTIME / "predictions" / "legacy_comparison" / "GBM_CPA_scGPT_PW034_Panobinostat_pred.h5ad")
    parser.add_argument("--target-patient", default="PW034")
    parser.add_argument("--target-drug", default="Panobinostat")
    parser.add_argument("--target-dosage", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    return parser.parse_args()


def dense(x): return x.toarray() if sparse.issparse(x) else np.asarray(x)


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
    def forward(self, x): return self.net(x)


def install_cpa_compat() -> type:
    from scvi.model._utils import parse_device_args
    from scvi import settings as scvi_settings
    import scvi.model._utils as scvi_model_utils
    import scvi.train as scvi_train
    import scvi.train._callbacks as scvi_callbacks
    from lightning.pytorch.callbacks import Callback
    from scvi.train import TrainRunner as CurrentTrainRunner

    def parse_use_gpu_arg(use_gpu=None, return_device=False):
        if use_gpu is None or use_gpu == "auto":
            accelerator, devices = "auto", "auto"
        elif use_gpu is False:
            accelerator, devices = "cpu", "auto"
        elif use_gpu is True:
            accelerator, devices = "cuda", "auto"
        elif isinstance(use_gpu, int):
            accelerator, devices = "cuda", [use_gpu]
        elif isinstance(use_gpu, str):
            value = use_gpu.lower()
            if value in {"cuda", "gpu"}: accelerator, devices = "cuda", "auto"
            elif value == "cpu": accelerator, devices = "cpu", "auto"
            else: accelerator, devices = "cuda", [int(use_gpu)]
        else: accelerator, devices = "auto", "auto"
        parsed = parse_device_args(accelerator=accelerator, devices=devices, return_device="torch")
        return parsed if return_device else parsed[:2]

    class SaveBestState(Callback):
        def __init__(self, monitor="validation_loss", mode="min", period=1, verbose=False, **kwargs):
            super().__init__()
            self.monitor = monitor; self.mode = mode; self.period = period; self.verbose = verbose

    if not hasattr(scvi_settings, "dl_pin_memory_gpu_training"):
        scvi_settings.dl_pin_memory_gpu_training = False
    scvi_model_utils.parse_use_gpu_arg = parse_use_gpu_arg
    scvi_callbacks.SaveBestState = SaveBestState

    class CompatTrainRunner(CurrentTrainRunner):
        def __init__(self, model, training_plan, data_splitter, max_epochs,
                     accelerator=None, devices=None, use_gpu=None, **trainer_kwargs):
            if accelerator is None:
                accelerator, parsed_devices = parse_use_gpu_arg(use_gpu=use_gpu, return_device=False)
                devices = parsed_devices if devices is None else devices
            if devices is None: devices = "auto"
            super().__init__(model=model, training_plan=training_plan, data_splitter=data_splitter,
                             max_epochs=max_epochs, accelerator=accelerator, devices=devices, **trainer_kwargs)
    scvi_train.TrainRunner = CompatTrainRunner

    def patch_lightning2_epoch_hooks(training_plan_cls):
        if getattr(training_plan_cls, "_codex_lightning2_compat", False): return
        old_training_step = training_plan_cls.training_step
        old_validation_step = training_plan_cls.validation_step
        old_training_epoch_end = training_plan_cls.training_epoch_end
        old_validation_epoch_end = training_plan_cls.validation_epoch_end
        def training_step(self, *args, **kwargs):
            output = old_training_step(self, *args, **kwargs)
            self._codex_train_outputs.append(output); return output
        def validation_step(self, *args, **kwargs):
            output = old_validation_step(self, *args, **kwargs)
            self._codex_validation_outputs.append(output); return output
        def on_train_epoch_start(self): self._codex_train_outputs = []
        def on_validation_epoch_start(self): self._codex_validation_outputs = []
        def on_train_epoch_end(self):
            outputs = getattr(self, "_codex_train_outputs", [])
            if outputs: old_training_epoch_end(self, outputs)
            self._codex_train_outputs = []
        def on_validation_epoch_end(self):
            outputs = getattr(self, "_codex_validation_outputs", [])
            if outputs: old_validation_epoch_end(self, outputs)
            self._codex_validation_outputs = []
        training_plan_cls.training_step = training_step
        training_plan_cls.validation_step = validation_step
        training_plan_cls.on_train_epoch_start = on_train_epoch_start
        training_plan_cls.on_validation_epoch_start = on_validation_epoch_start
        training_plan_cls.on_train_epoch_end = on_train_epoch_end
        training_plan_cls.on_validation_epoch_end = on_validation_epoch_end
        delattr(training_plan_cls, "training_epoch_end")
        delattr(training_plan_cls, "validation_epoch_end")
        training_plan_cls._codex_lightning2_compat = True

    spec = importlib.util.find_spec("cpa")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    package = types.ModuleType("cpa")
    package.__path__ = [str(package_dir)]
    sys.modules["cpa"] = package
    model_spec = importlib.util.spec_from_file_location("cpa._model", package_dir / "_model.py")
    model_module = importlib.util.module_from_spec(model_spec)
    sys.modules["cpa._model"] = model_module
    model_spec.loader.exec_module(model_module)
    from cpa._task import CPATrainingPlan
    patch_lightning2_epoch_hooks(CPATrainingPlan)
    return model_module.CPA


def reset_cpa_class_state(CPA): CPA.pert_encoder = None; CPA.covars_encoder = None; CPA.pert_smiles_map = None


def build_aligned_adata(adata, aligner_path, scgpt_key, device):
    """Project scGPT embeddings through trained aligner → create new AnnData with aligned X."""
    ckpt = torch.load(aligner_path, map_location="cpu")
    model = Aligner(input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    X_scgpt = adata.obsm[scgpt_key].astype(np.float32)
    n_cells = X_scgpt.shape[0]
    all_aligned = np.zeros((n_cells, ckpt["output_dim"]), dtype=np.float32)
    bs = 512
    with torch.no_grad():
        for start in range(0, n_cells, bs):
            end = min(start + bs, n_cells)
            batch = torch.tensor(X_scgpt[start:end], dtype=torch.float32).to(device)
            all_aligned[start:end] = model(batch).cpu().numpy()
    all_aligned = np.clip(all_aligned, 0, None)
    print(f"  Aligned: {all_aligned.shape}, min={all_aligned.min():.4f}, max={all_aligned.max():.4f}")

    # Create new AnnData with aligned values as X and as a counts-like layer
    new_adata = ad.AnnData(
        X=all_aligned,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=adata.uns.copy(),
    )
    new_adata.layers["counts"] = all_aligned.copy()
    new_adata.obsm = {k: v.copy() for k, v in adata.obsm.items()}
    return new_adata


def main():
    args = parse_args()
    log_path = args.predicted.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log_handle, \
         contextlib.redirect_stdout(Tee(sys.stdout, log_handle)), \
         contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Method: {args.method}, scGPT key: {args.scgpt_key}")
        print(f"Device: {device}")

        # Load aligner and project scGPT → gene space
        print("Building aligned AnnData...")
        adata_raw = ad.read_h5ad(args.adata)
        adata_raw.obs = adata_raw.obs.copy()
        adata_raw.obs["perturbation"] = adata_raw.obs["perturbation"].astype(str)
        adata_raw.obs["covariate_patient"] = adata_raw.obs["covariate_patient"].astype(str)
        adata_raw.obs["cell_type"] = adata_raw.obs["cell_type"].astype(str)
        adata_raw.obs["dosage"] = pd.to_numeric(adata_raw.obs["dosage"], errors="raise").astype(float).astype(str)

        adata = build_aligned_adata(adata_raw, args.aligner, args.scgpt_key, device)
        del adata_raw

        # Install CPA
        CPA = install_cpa_compat()

        # Setup CPA (use gauss loss since aligned values are continuous, not counts)
        reset_cpa_class_state(CPA)
        CPA.setup_anndata(
            adata,
            perturbation_key="perturbation",
            control_group="control",
            dosage_key="dosage",
            categorical_covariate_keys=["covariate_patient"],
            layer="counts",
            is_count_data=False,
        )

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        use_gpu = False if args.device == "cpu" else ("auto" if args.device == "auto" else True)
        model = CPA(
            adata,
            split_key="split",
            train_split="train",
            valid_split="valid",
            test_split="ood",
            n_latent=32,
            recon_loss="gauss",
            seed=args.seed,
        )

        if args.model_dir.exists():
            backup = args.model_dir.with_name(f"{args.model_dir.name}.previous")
            if backup.exists(): shutil.rmtree(backup)
            args.model_dir.rename(backup)

        print(f"Training {args.method}...")
        model.train(
            max_epochs=args.max_epochs,
            use_gpu=use_gpu,
            batch_size=args.batch_size,
            save_path=str(args.model_dir),
            check_val_every_n_epoch=1,
            early_stopping_patience=args.early_stopping_patience,
            plan_kwargs={"do_clip_grad": True, "gradient_clip_value": 3.0, "n_epochs_verbose": 1},
            log_every_n_steps=25,
            enable_progress_bar=True,
        )

        # Save training history
        history = model.epoch_history.copy()
        args.model_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.model_dir / "history.csv", index=False)
        history.to_csv(args.model_dir / "epoch_history.tsv", sep="\t", index=False)

        valid = history[history["mode"].eq("valid")].copy()
        if not valid.empty:
            valid["cpa_metric"] = valid["r2_mean"] + 0.5 * valid["r2_var"] + np.exp(valid["disnt_after"] - valid["disnt_basal"])
            best = valid.loc[valid["cpa_metric"].idxmax()]
            print(f"Best epoch {int(best['epoch'])}: r2_mean={best['r2_mean']:.4f}, r2_var={best['r2_var']:.4f}")

        # Generate OOD predictions
        from cpa._utils import CPA_REGISTRY_KEYS
        basal_mask = (adata.obs["covariate_patient"].eq(args.target_patient) & adata.obs["perturbation"].eq("control")).to_numpy()
        basal = adata[basal_mask].copy()
        basal.obs["perturbation"] = args.target_drug
        basal.obs["dosage"] = str(float(args.target_dosage))
        basal.obs["is_control"] = False
        basal.obs["split"] = "ood_predict"
        basal.obs["CPA_control"] = 0

        max_comb_len = int(CPA_REGISTRY_KEYS.MAX_COMB_LENGTH)
        perts = [model.pert_encoder[args.target_drug]] + [CPA_REGISTRY_KEYS.PADDING_IDX] * (max_comb_len - 1)
        doses = [float(args.target_dosage)] + [0.0] * (max_comb_len - 1)
        basal.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS] = np.tile(np.asarray(perts, dtype=np.int64), (basal.n_obs, 1))
        basal.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS_DOSAGES] = np.tile(np.asarray(doses, dtype=np.float32), (basal.n_obs, 1))

        category_key = CPA_REGISTRY_KEYS.CATEGORY_KEY
        if category_key in basal.obs:
            basal.obs[category_key] = basal.obs[["covariate_patient", "perturbation"]].apply(lambda row: "_".join(row.astype(str)), axis=1)

        model.predict(basal, batch_size=args.batch_size, n_samples=1, return_mean=True)
        pred_x = np.asarray(basal.obsm["CPA_pred"], dtype=np.float32)
        pred_x = np.nan_to_num(pred_x, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
        pred_x[pred_x < 0] = 0.0

        pred = ad.AnnData(X=pred_x, obs=basal.obs.copy(), var=adata.var.copy())
        pred.obs_names = basal.obs_names.copy()
        pred.write_h5ad(args.predicted, compression="gzip")
        print(f"Prediction saved: {args.predicted}, shape={pred.shape}")
        print("Done.")


if __name__ == "__main__":
    main()
