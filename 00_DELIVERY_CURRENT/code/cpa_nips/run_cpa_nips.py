#!/usr/bin/env python
"""Run CPA on NIPS-Ready GBM dataset and evaluate with NIPS protocol.

This script:
  1. Trains CPA on GBM_NIPS_Ready.h5ad
  2. Predicts OOD perturbation (PW034/Panobinostat)
  3. Evaluates with NIPS multi-group protocol (evaluate_nips_gbm.py)

Usage:
  conda activate plknature
  python 00_DELIVERY_CURRENT/code/cpa_nips/run_cpa_nips.py
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
from lightning.pytorch.loggers import TensorBoardLogger
from scipy import sparse, stats
from sklearn.metrics import r2_score

from nips_aliases import (
    control_mask,
    deg_genes,
    ensure_nips_aliases,
    group_name,
    set_counterfactual_obs,
)

ROOT = Path(__file__).resolve().parents[3]
DELIVERY = ROOT / "00_DELIVERY_CURRENT"

_ALLOWED_CUDA_VISIBLE_DEVICES = {"0", "1", "2", "3", "4", "5", "6", "7"}
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
_requested_cuda_visible_devices = {
    item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()
}
if _requested_cuda_visible_devices and not _requested_cuda_visible_devices.issubset(_ALLOWED_CUDA_VISIBLE_DEVICES):
    raise RuntimeError(
        "This CPA runner is restricted to physical GPUs 0-7. "
        "Set CUDA_VISIBLE_DEVICES accordingly."
    )


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

    def close(self) -> None:
        for stream in self.streams:
            if hasattr(stream, "close"):
                stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata", type=Path, default=DELIVERY / "dataset" / "GBM_NIPS_Ready.h5ad")
    parser.add_argument("--model-dir", type=Path, default=DELIVERY / "models" / "GBM_CPA_NIPS_model")
    parser.add_argument("--predicted", type=Path, default=DELIVERY / "predictions" / "GBM_CPA_NIPS_PW034_Panobinostat_pred.h5ad")
    parser.add_argument("--training-log", type=Path, default=DELIVERY / "logs" / "GBM_CPA_NIPS_training.log")
    parser.add_argument("--output-dir", type=Path, default=DELIVERY / "evaluation" / "nips")
    parser.add_argument("--target-patient", "--target-cell-type", dest="target_patient", default="PW034")
    parser.add_argument("--target-drug", "--target-condition", dest="target_drug", default="Panobinostat")
    parser.add_argument("--target-dosage", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sinkhorn-samples", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--skip-eval", action="store_true", help="Skip NIPS evaluation after training")
    return parser.parse_args()


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True))


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
            if value in {"cuda", "gpu"}:
                accelerator, devices = "cuda", "auto"
            elif value == "cpu":
                accelerator, devices = "cpu", "auto"
            else:
                accelerator, devices = "cuda", [int(use_gpu)]
        else:
            accelerator, devices = "auto", "auto"

        parsed = parse_device_args(
            accelerator=accelerator,
            devices=devices,
            return_device="torch",
        )
        return parsed if return_device else parsed[:2]

    class SaveBestState(Callback):
        def __init__(self, monitor="validation_loss", mode="min", period=1, verbose=False, **kwargs):
            super().__init__()
            self.monitor = monitor
            self.mode = mode
            self.period = period
            self.verbose = verbose

    if not hasattr(scvi_settings, "dl_pin_memory_gpu_training"):
        scvi_settings.dl_pin_memory_gpu_training = False
    scvi_model_utils.parse_use_gpu_arg = parse_use_gpu_arg
    scvi_callbacks.SaveBestState = SaveBestState

    class CompatTrainRunner(CurrentTrainRunner):
        def __init__(
            self,
            model,
            training_plan,
            data_splitter,
            max_epochs,
            accelerator=None,
            devices=None,
            use_gpu=None,
            **trainer_kwargs,
        ):
            if accelerator is None:
                accelerator, parsed_devices = parse_use_gpu_arg(use_gpu=use_gpu, return_device=False)
                devices = parsed_devices if devices is None else devices
            if devices is None:
                devices = "auto"
            super().__init__(
                model=model,
                training_plan=training_plan,
                data_splitter=data_splitter,
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices,
                **trainer_kwargs,
            )

    scvi_train.TrainRunner = CompatTrainRunner

    def patch_lightning2_epoch_hooks(training_plan_cls):
        if getattr(training_plan_cls, "_codex_lightning2_compat", False):
            return
        old_training_step = training_plan_cls.training_step
        old_validation_step = training_plan_cls.validation_step
        old_training_epoch_end = training_plan_cls.training_epoch_end
        old_validation_epoch_end = training_plan_cls.validation_epoch_end

        def training_step(self, *args, **kwargs):
            output = old_training_step(self, *args, **kwargs)
            self._codex_train_outputs.append(output)
            return output

        def validation_step(self, *args, **kwargs):
            output = old_validation_step(self, *args, **kwargs)
            self._codex_validation_outputs.append(output)
            return output

        def on_train_epoch_start(self):
            self._codex_train_outputs = []

        def on_validation_epoch_start(self):
            self._codex_validation_outputs = []

        def on_train_epoch_end(self):
            outputs = getattr(self, "_codex_train_outputs", [])
            if outputs:
                old_training_epoch_end(self, outputs)
            self._codex_train_outputs = []

        def on_validation_epoch_end(self):
            outputs = getattr(self, "_codex_validation_outputs", [])
            if outputs:
                old_validation_epoch_end(self, outputs)
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
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Could not locate installed cpa package")

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    package = types.ModuleType("cpa")
    package.__path__ = [str(package_dir)]
    sys.modules["cpa"] = package

    model_spec = importlib.util.spec_from_file_location("cpa._model", package_dir / "_model.py")
    if model_spec is None or model_spec.loader is None:
        raise ImportError(f"Could not load CPA model module from {package_dir}")
    model_module = importlib.util.module_from_spec(model_spec)
    sys.modules["cpa._model"] = model_module
    model_spec.loader.exec_module(model_module)
    from cpa._task import CPATrainingPlan

    patch_lightning2_epoch_hooks(CPATrainingPlan)
    return model_module.CPA


def validate_adata(adata: ad.AnnData, patient: str, drug: str) -> None:
    required_obs = {"condition", "dose_val", "cell_type", "split", "neg_control", "cov_drug_name"}
    missing = sorted(required_obs.difference(adata.obs.columns))
    if missing:
        raise ValueError(f"Missing required obs columns: {missing}")
    if "counts" not in adata.layers:
        raise ValueError("Expected integer counts in adata.layers['counts']")
    genes = deg_genes(adata, patient, drug)

    counts = adata.layers["counts"]
    library = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(~np.isfinite(library)) or np.any(library <= 0):
        n_bad = int((~np.isfinite(library) | (library <= 0)).sum())
        raise ValueError(f"Found {n_bad} cells with non-positive/non-finite count library sums")

    split_counts = adata.obs["split"].value_counts().to_dict()
    for split in ("train", "valid", "ood"):
        if split_counts.get(split, 0) <= 0:
            raise ValueError(f"Split '{split}' has no cells: {split_counts}")

    ctrl = control_mask(adata.obs)
    n_basal = int(((adata.obs["cell_type"] == patient).to_numpy() & ctrl).sum())
    n_true = int((adata.obs["cov_drug_name"].astype(str) == group_name(patient, drug)).sum())
    if n_basal <= 0 or n_true <= 0:
        raise ValueError(f"Missing basal or true cells for {patient}|{drug}: basal={n_basal}, true={n_true}")

    emit(
        "adata_validated",
        shape=list(adata.shape),
        split_counts=split_counts,
        n_basal=n_basal,
        n_true=n_true,
        degs=len(genes),
    )

def reset_cpa_class_state(CPA: type) -> None:
    CPA.pert_encoder = None
    CPA.covars_encoder = None
    CPA.pert_smiles_map = None


def train_cpa(CPA: type, adata: ad.AnnData, args: argparse.Namespace):
    reset_cpa_class_state(CPA)
    CPA.setup_anndata(
        adata,
        perturbation_key="condition",
        control_group="control",
        dosage_key="dose_val",
        categorical_covariate_keys=["cell_type"],
        layer="counts",
        is_count_data=True,
    )
    emit("setup_complete", pert_encoder=CPA.pert_encoder, covars_encoder=CPA.covars_encoder)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    use_gpu = False if args.device == "cpu" else ("auto" if args.device == "auto" else True)
    model = CPA(
        adata,
        split_key="split",
        train_split="train",
        valid_split="valid",
        test_split="ood",
        n_latent=32,
        recon_loss="nb",
        seed=args.seed,
    )

    if args.model_dir.exists():
        backup = args.model_dir.with_name(f"{args.model_dir.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        args.model_dir.rename(backup)
        emit("model_dir_backed_up", backup=str(backup))

    tb_log_dir = args.model_dir / "tensorboard_logs"
    tb_logger = TensorBoardLogger(save_dir=str(tb_log_dir), name="cpa_training")

    emit(
        "train_start",
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        tensorboard_log_dir=str(tb_logger.log_dir),
    )
    model.train(
        max_epochs=args.max_epochs,
        use_gpu=use_gpu,
        batch_size=args.batch_size,
        save_path=str(args.model_dir),
        check_val_every_n_epoch=1,
        early_stopping_patience=args.early_stopping_patience,
        plan_kwargs={
            "do_clip_grad": True,
            "gradient_clip_value": 3.0,
            "n_epochs_verbose": 1,
        },
        log_every_n_steps=25,
        enable_progress_bar=True,
        logger=tb_logger,
    )

    history = model.epoch_history.copy()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.model_dir / "history.csv", index=False)
    history.to_csv(args.model_dir / "epoch_history.tsv", sep="\t", index=False)
    valid = history[history["mode"].eq("valid")].copy()
    best = {}
    if not valid.empty:
        valid["cpa_metric"] = valid["r2_mean"] + 0.5 * valid["r2_var"] + np.exp(valid["disnt_after"] - valid["disnt_basal"])
        best_row = valid.loc[valid["cpa_metric"].idxmax()]
        best = {
            "epoch": int(best_row["epoch"]),
            "val_recon": float(best_row["recon_loss"]),
            "val_r2_mean": float(best_row["r2_mean"]),
            "val_r2_var": float(best_row["r2_var"]),
            "cpa_metric": float(best_row["cpa_metric"]),
        }
    emit("train_complete", epochs_logged=int(history["epoch"].nunique()), best_valid=best)
    print(history.tail(10).to_string(index=False))
    return model


def predict_ood(model, adata: ad.AnnData, args: argparse.Namespace) -> ad.AnnData:
    basal_mask = (
        adata.obs["cell_type"].eq(args.target_patient).to_numpy()
        & control_mask(adata.obs)
    )
    basal = adata[basal_mask].copy()
    emit("basal_extracted", n=int(basal.n_obs))

    set_counterfactual_obs(
        basal.obs,
        cell_type=args.target_patient,
        condition=args.target_drug,
        dose=float(args.target_dosage),
    )
    basal.obs["split"] = "ood_predict"
    basal.obs["CPA_control"] = 0

    from cpa._utils import CPA_REGISTRY_KEYS

    max_comb_len = int(CPA_REGISTRY_KEYS.MAX_COMB_LENGTH)
    perts = [model.pert_encoder[args.target_drug]] + [CPA_REGISTRY_KEYS.PADDING_IDX] * (max_comb_len - 1)
    doses = [float(args.target_dosage)] + [0.0] * (max_comb_len - 1)
    basal.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS] = np.tile(np.asarray(perts, dtype=np.int64), (basal.n_obs, 1))
    basal.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS_DOSAGES] = np.tile(np.asarray(doses, dtype=np.float32), (basal.n_obs, 1))

    category_key = CPA_REGISTRY_KEYS.CATEGORY_KEY
    if category_key in basal.obs:
        basal.obs[category_key] = basal.obs[["cell_type", "condition"]].apply(lambda row: "_".join(row.astype(str)), axis=1)

    model.predict(basal, batch_size=args.batch_size, n_samples=1, return_mean=True)
    pred_x = np.asarray(basal.obsm["CPA_pred"], dtype=np.float32)
    pred_x = np.nan_to_num(pred_x, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    pred_x[pred_x < 0] = 0.0

    pred = ad.AnnData(X=pred_x, obs=basal.obs.copy(), var=adata.var.copy(), uns={"prediction": {
        "method": "CPA",
        "target_patient": args.target_patient,
        "target_drug": args.target_drug,
        "target_dosage": float(args.target_dosage),
        "x_semantics": "CPA predicted mean expression from model.predict using recon_loss='nb'.",
        "dataset": "NIPS-compatible",
    }})
    pred.obs_names = basal.obs_names.copy()
    args.predicted.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(args.predicted, compression="gzip")
    emit(
        "prediction_saved",
        output=str(args.predicted),
        shape=list(pred.shape),
        mean=float(pred_x.mean()),
        max=float(pred_x.max()),
    )
    return pred

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


def compute_metrics(adata: ad.AnnData, pred: ad.AnnData, args: argparse.Namespace) -> dict[str, float | str]:
    genes = deg_genes(adata, args.target_patient, args.target_drug)
    gene_idx = adata.var_names.get_indexer(genes)
    pred_idx = pred.var_names.get_indexer(genes)
    if np.any(gene_idx < 0) or np.any(pred_idx < 0):
        raise ValueError("DEG genes are missing from reference or prediction")

    true_mask = (
        adata.obs["cell_type"].eq(args.target_patient)
        & adata.obs["condition"].eq(args.target_drug)
    ).to_numpy()
    ctrl_mask = (
        adata.obs["cell_type"].eq(args.target_patient).to_numpy()
        & control_mask(adata.obs)
    )

    x_true = dense(adata.X[true_mask][:, gene_idx])
    x_ctrl = dense(adata.X[ctrl_mask][:, gene_idx])
    x_pred = dense(pred.X[:, pred_idx])

    true_post = x_true.mean(axis=0)
    pred_post = x_pred.mean(axis=0)
    ctrl = x_ctrl.mean(axis=0)
    true_logfc = true_post - ctrl
    pred_logfc = pred_post - ctrl

    metrics = {
        "method": "CPA",
        "patient": args.target_patient,
        "drug": args.target_drug,
        "pearson": float(stats.pearsonr(pred_logfc, true_logfc).statistic),
        "spearman": float(stats.spearmanr(pred_logfc, true_logfc).statistic),
        "r2": float(r2_score(true_post, pred_post)),
        "sinkhorn": sinkhorn_distance(x_pred, x_true, args.sinkhorn_samples, args.seed),
        "direction": float(np.mean(np.sign(pred_logfc) == np.sign(true_logfc)) * 100.0),
    }
    emit("metrics_computed", group=group_name(args.target_patient, args.target_drug), **metrics)
    return metrics

def run_nips_evaluation(adata_path: Path, pred_path: Path, method: str, output_dir: Path, setting: str = "ood"):
    """Run NIPS evaluation protocol."""
    import subprocess

    eval_script = DELIVERY / "code" / "cpa_nips" / "evaluate_nips_gbm.py"
    if not eval_script.exists():
        print(f"WARNING: NIPS evaluation script not found at {eval_script}")
        return

    cmd = [
        sys.executable, str(eval_script),
        "--adata", str(adata_path),
        "--predicted", str(pred_path),
        "--method", method,
        "--setting", setting,
        "--output-dir", str(output_dir),
    ]
    print(f"\nRunning NIPS evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"NIPS evaluation returned non-zero exit code: {result.returncode}")
    else:
        print("NIPS evaluation completed successfully.")


def main() -> None:
    args = parse_args()
    args.training_log.parent.mkdir(parents=True, exist_ok=True)
    with args.training_log.open("w") as log_handle, contextlib.redirect_stdout(Tee(sys.stdout, log_handle)), contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):
        emit("run_start", args={k: str(v) for k, v in vars(args).items()})
        os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

        # Step 1: Load CPA
        CPA = install_cpa_compat()
        emit("cpa_loaded", cls=f"{CPA.__module__}.{CPA.__name__}")

        # Step 2: Load NIPS dataset
        print(f"\n{'='*60}")
        print(f"Loading NIPS dataset: {args.adata}")
        print(f"{'='*60}")
        adata = ad.read_h5ad(args.adata)
        adata.obs = adata.obs.copy()
        ensure_nips_aliases(adata, add_legacy_aliases=True)

        validate_adata(adata, args.target_patient, args.target_drug)

        # Step 3: Train CPA
        print(f"\n{'='*60}")
        print(f"Training CPA model")
        print(f"{'='*60}")
        model = train_cpa(CPA, adata, args)

        # Step 4: Predict OOD
        print(f"\n{'='*60}")
        print(f"Predicting OOD: {args.target_patient}/{args.target_drug}")
        print(f"{'='*60}")
        pred = predict_ood(model, adata, args)

        # Step 5: Compute metrics
        print(f"\n{'='*60}")
        print(f"Computing metrics")
        print(f"{'='*60}")
        metrics = compute_metrics(adata, pred, args)

        # Print summary
        print(f"\n{'='*60}")
        print(f"CPA NIPS Experiment Summary")
        print(f"{'='*60}")
        print(f"Method: CPA")
        print(f"Patient: {args.target_patient}")
        print(f"Drug: {args.target_drug}")
        print(f"Pearson (DE): {metrics['pearson']:.4f}")
        print(f"Spearman (DE): {metrics['spearman']:.4f}")
        print(f"R2 (DE): {metrics['r2']:.4f}")
        print(f"Sinkhorn: {metrics['sinkhorn']:.4f}")
        print(f"Direction: {metrics['direction']:.1f}%")

        # Step 6: Run NIPS evaluation
        if not args.skip_eval:
            print(f"\n{'='*60}")
            print(f"Running NIPS evaluation protocol")
            print(f"{'='*60}")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            run_nips_evaluation(args.adata, args.predicted, "CPA_NIPS", args.output_dir)

        emit("run_complete")


if __name__ == "__main__":
    main()
