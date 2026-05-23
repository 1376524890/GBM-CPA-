#!/usr/bin/env python
"""Run CPA (M0/M1/M4/M5) on the original NIPS Perturb-seq dataset.

NIPS dataset: 240,059 cells x 18,211 genes
  - Cell types: T cells CD4+, NK cells, Myeloid cells, T cells CD8+, B cells, T regulatory cells
  - OOD cell types (split): Myeloid cells, T regulatory cells
  - Drugs: ~180 compounds
  - DEGs: 599 groups in rank_genes_groups_cov

Variants:
  M0: raw counts, learnable drug embedding, NB loss
  M1: scGPT 512d -> Aligner MLP -> 18211d gene space, learnable drug embedding, Gauss loss
  M4: raw counts, MolFormer 768d frozen drug embedding, NB loss
  M5: scGPT -> Aligner -> gene space, MolFormer frozen drug embedding, Gauss loss

Usage:
  conda activate plknature
  # Run all 4 variants sequentially:
  python run_cpa_nips_dataset.py --variant all

  # Run a single variant:
  python run_cpa_nips_dataset.py --variant M0
  python run_cpa_nips_dataset.py --variant M1
  python run_cpa_nips_dataset.py --variant M4
  python run_cpa_nips_dataset.py --variant M5

  # Custom options:
  python run_cpa_nips_dataset.py --variant M0 --max-epochs 100 --batch-size 2048

  # Multi-GPU (must specify --strategy and --devices explicitly):
  python run_cpa_nips_dataset.py --variant M0 --devices 0,1 --strategy ddp_find_unused_parameters_true

  # Force fresh run (clear progress, ignore checkpoints):
  python run_cpa_nips_dataset.py --variant M1 --fresh
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes.util
import importlib.util
import json
import os
import shutil
import sys
import sysconfig
import types
from pathlib import Path
from typing import Any, TextIO

import anndata as ad
import numpy as np
import pandas as pd
from lightning.pytorch.loggers import TensorBoardLogger

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
from scipy import sparse, stats
from sklearn.metrics import mean_squared_error as mse_sklearn

ROOT = Path(__file__).resolve().parents[3]
DELIVERY = ROOT / "00_DELIVERY_CURRENT"

NIPS_DATA_DIR = Path("/home/u2023312303/nature子刊/zyq/data")
DEFAULT_NIPS_H5AD = NIPS_DATA_DIR / "nips_pp_scFM_MolFormer.h5ad"
DEFAULT_MOLFORMER_PARQUET = NIPS_DATA_DIR / "nips_molformer_drug_emb.parquet"

_ALLOWED_CUDA_VISIBLE_DEVICES = {"0", "1", "2", "3", "4", "5", "6", "7"}
_requested = {s.strip() for s in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if s.strip()}
if _requested and not _requested.issubset(_ALLOWED_CUDA_VISIBLE_DEVICES):
    raise RuntimeError("Restricted to physical GPUs 0-7.")


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams
    def write(self, text: str) -> int:
        for s in self.streams: s.write(text); s.flush()
        return len(text)
    def flush(self):
        for s in self.streams: s.flush()
    def close(self):
        for s in self.streams:
            if s not in (sys.stdout, sys.stderr) and hasattr(s, "close"):
                s.close()


def parse_args():
    p = argparse.ArgumentParser(description="CPA variants on NIPS dataset")
    p.add_argument("--adata", type=Path, default=DEFAULT_NIPS_H5AD)
    p.add_argument("--molformer-parquet", type=Path, default=DEFAULT_MOLFORMER_PARQUET)
    p.add_argument("--variant", default="all", choices=["M0", "M1", "M4", "M5", "all"],
                   help="CPA variant to run, or 'all' for sequential")
    p.add_argument("--split-key", default="split", choices=["split", "split2", "split3"])
    p.add_argument("--eval-settings", nargs="+", default=["ood", "test"])
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-latent", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--devices", default="0",
                   help="Lightning CUDA devices, e.g. '0,1', '0', or 'auto'.")
    p.add_argument("--strategy", default="auto",
                   help="Lightning strategy. Use 'auto' to let Lightning choose. CUDA DDP must not use ddp_fork.")
    p.add_argument("--fresh", action="store_true",
                   help="Force fresh run: clear progress, ignore existing checkpoints and cached data.")
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="DataLoader worker processes per rank.")
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="DataLoader prefetch_factor when num_workers > 0.")
    p.add_argument("--no-persistent-workers", action="store_true",
                   help="Disable persistent DataLoader workers.")
    p.add_argument("--min-treated", type=int, default=5)
    p.add_argument("--min-ctrl", type=int, default=5)
    p.add_argument("--min-deg", type=int, default=2)
    p.add_argument("--sinkhorn-samples", type=int, default=512)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)

def emit(event, **kw):
    print(json.dumps({"event": event, **kw}, ensure_ascii=False, sort_keys=True))


# ==============================================================================
# Resume / checkpoint helpers
# ==============================================================================

def _progress_path(variant: str) -> Path:
    return DELIVERY / "models" / f"CPA_NIPS_{variant}_progress.json"


def load_progress(variant: str) -> dict:
    p = _progress_path(variant)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def save_progress(variant: str, progress: dict) -> None:
    p = _progress_path(variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(progress, indent=2))


def mark_done(variant: str, stage: str) -> None:
    prog = load_progress(variant)
    prog[stage] = True
    save_progress(variant, prog)


def clear_progress(variants: list[str]) -> None:
    """Remove progress files and aligned caches for a fresh run."""
    for v in variants:
        p = _progress_path(v)
        if p.exists():
            p.unlink()
            emit("progress_cleared", variant=v)
        # Also remove aligned adata cache so it gets rebuilt
        aligned_cache = DELIVERY / "models" / f"CPA_NIPS_{v}_aligned.h5ad"
        if aligned_cache.exists():
            aligned_cache.unlink()
            emit("aligned_cache_cleared", variant=v)


def is_done(variant: str, stage: str) -> bool:
    return load_progress(variant).get(stage, False)


def find_last_checkpoint(model_dir: Path) -> str | None:
    """Find the last Lightning checkpoint in model_dir for resume."""
    # scvi-tools saves checkpoints in model_dir/checkpoints/
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1])
    # Also check for last.ckpt directly in model_dir
    last = model_dir / "last.ckpt"
    if last.exists():
        return str(last)
    # Check for any .ckpt file
    ckpts = sorted(model_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        return str(ckpts[-1])
    return None


def _aligner_path(variant: str) -> Path:
    return DELIVERY / "models" / f"CPA_NIPS_{variant}_aligner.pt"


def _symlink_force(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.is_symlink() or dst.exists():
        try:
            if dst.resolve() == src.resolve():
                return
        except OSError:
            pass
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def configure_keops_cuda() -> bool:
    """Make PyKeOps see CUDA libs shipped inside the active conda env."""
    if getattr(configure_keops_cuda, "_configured", False):
        return bool(getattr(configure_keops_cuda, "_available", False))

    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = purelib / "nvidia"
    runtime = nvidia_root / "cuda_runtime"
    nvrtc = nvidia_root / "cuda_nvrtc"
    cccl = nvidia_root / "cuda_cccl"
    libcuda = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")

    required = [
        runtime / "include" / "cuda.h",
        nvrtc / "include" / "nvrtc.h",
        runtime / "lib" / "libcudart.so.12",
        nvrtc / "lib" / "libnvrtc.so.12",
        libcuda,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        emit("keops_cuda_unavailable", reason="missing_cuda_files", missing=missing)
        configure_keops_cuda._configured = True
        configure_keops_cuda._available = False
        return False

    shim = DELIVERY / ".keops_cuda12_shim"
    include_dir = shim / "include"
    lib_dir = shim / "lib"
    include_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    for src_dir in (runtime / "include", nvrtc / "include", cccl / "include"):
        if not src_dir.exists():
            continue
        for child in src_dir.iterdir():
            _symlink_force(child, include_dir / child.name)

    _symlink_force(nvrtc / "lib" / "libnvrtc.so.12", lib_dir / "libnvrtc.so")
    for builtins in (nvrtc / "lib").glob("libnvrtc-builtins.so*"):
        _symlink_force(builtins, lib_dir / builtins.name)
    _symlink_force(runtime / "lib" / "libcudart.so.12", lib_dir / "libcudart.so")
    _symlink_force(libcuda, lib_dir / "libcuda.so")

    os.environ.setdefault("CUDA_PATH", str(shim))
    os.environ.setdefault("CUDA_HOME", str(shim))

    original_find_library = ctypes.util.find_library

    def find_library_patched(name):
        shim_libs = {
            "cuda": lib_dir / "libcuda.so",
            "nvrtc": lib_dir / "libnvrtc.so",
            "cudart": lib_dir / "libcudart.so",
        }
        if name in shim_libs and shim_libs[name].exists():
            return str(shim_libs[name])
        return original_find_library(name)

    ctypes.util.find_library = find_library_patched
    configure_keops_cuda._configured = True
    configure_keops_cuda._available = True
    emit("keops_cuda_configured", cuda_path=str(shim), lib_path=str(lib_dir))
    return True


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def process_rank() -> int:
    for name in ("RANK", "GLOBAL_RANK", "LOCAL_RANK"):
        if name in os.environ:
            return _env_int(name)
    return 0


def is_primary_process() -> bool:
    if "RANK" in os.environ:
        return _env_int("RANK") == 0
    if "GLOBAL_RANK" in os.environ:
        return _env_int("GLOBAL_RANK") == 0
    return _env_int("NODE_RANK") == 0 and _env_int("LOCAL_RANK") == 0


def parse_lightning_devices(value: str) -> Any:
    value = str(value).strip().lower()
    if value in {"auto", "all"}:
        return "auto"
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(value)]


def _requested_logical_cuda_ids(devices: Any) -> list[int] | None:
    if devices == "auto":
        return None
    if isinstance(devices, int):
        return [devices]
    return list(devices)


def validate_cuda_devices(args) -> None:
    if args.device == "cpu":
        return

    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        if args.device == "cuda":
            raise RuntimeError(
                "CUDA was requested, but PyTorch reports no available CUDA device. "
                f"CUDA_VISIBLE_DEVICES={visible_env!r}."
            )
        return

    visible_count = torch.cuda.device_count()
    requested = _requested_logical_cuda_ids(parse_lightning_devices(args.devices))
    visible_ids = list(range(visible_count))
    emit(
        "cuda_visible",
        cuda_visible_devices=visible_env,
        torch_device_count=visible_count,
        torch_logical_devices=visible_ids,
    )
    if requested is None:
        return
    if not requested or min(requested) < 0 or max(requested) >= visible_count:
        raise RuntimeError(
            "Lightning devices must be logical CUDA ids visible to this Python process. "
            f"Requested --devices={args.devices!r}, but PyTorch only sees logical GPUs "
            f"{visible_ids} because CUDA_VISIBLE_DEVICES={visible_env!r}. "
            "For two GPUs, run with CUDA_VISIBLE_DEVICES=0,1 and --devices 0,1. "
            "For one GPU, use --devices 0."
        )


def lightning_train_kwargs(args) -> dict[str, Any]:
    if args.device == "cpu":
        return {"accelerator": "cpu", "devices": "auto"}
    strategy = args.strategy
    if strategy.startswith("ddp_fork"):
        strategy = strategy.replace("ddp_fork", "ddp", 1)
        emit("strategy_rewritten", requested=args.strategy, effective=strategy)
    kwargs: dict[str, Any] = {
        "accelerator": "cuda" if args.device == "cuda" else "auto",
        "devices": parse_lightning_devices(args.devices),
    }
    if strategy != "auto":
        kwargs["strategy"] = strategy
    return kwargs


def configure_cpa_dataloading(args) -> None:
    import cpa._data as cpa_data
    import cpa._model as cpa_model

    dl_kwargs: dict[str, Any] = {"num_workers": max(0, int(args.num_workers))}
    if dl_kwargs["num_workers"] > 0:
        dl_kwargs["persistent_workers"] = not args.no_persistent_workers
        dl_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    cpa_data.AnnDataSplitter._codex_data_loader_kwargs = dl_kwargs
    cpa_model.AnnDataSplitter._codex_data_loader_kwargs = dl_kwargs
    emit("dataloader_configured", **dl_kwargs)


def training_torch_device(args) -> str:
    if args.device == "cpu" or not torch.cuda.is_available():
        return "cpu"
    devices = parse_lightning_devices(args.devices)
    first = devices[0] if isinstance(devices, list) else (0 if devices == "auto" else devices)
    return f"cuda:{first}"


# ==============================================================================
# CPA compatibility patches
# ==============================================================================

def install_cpa_compat():
    from scvi.model._utils import parse_device_args
    from scvi import settings as scvi_settings
    import scvi.model._utils as scvi_model_utils
    import scvi.train as scvi_train
    import scvi.train._callbacks as scvi_callbacks
    from lightning.pytorch.callbacks import Callback
    from scvi.train import TrainRunner as CurrentTrainRunner

    def parse_use_gpu_arg(use_gpu=None, return_device=False):
        if use_gpu is None or use_gpu == "auto": a, d = "auto", "auto"
        elif use_gpu is False: a, d = "cpu", "auto"
        elif use_gpu is True: a, d = "cuda", "auto"
        elif isinstance(use_gpu, int): a, d = "cuda", [use_gpu]
        elif isinstance(use_gpu, str):
            v = use_gpu.lower()
            if v in {"cuda","gpu"}: a, d = "cuda", "auto"
            elif v == "cpu": a, d = "cpu", "auto"
            else: a, d = "cuda", [int(v)]
        else: a, d = "auto", "auto"
        parsed = parse_device_args(accelerator=a, devices=d, return_device="torch")
        return parsed if return_device else parsed[:2]

    class SaveBestState(Callback):
        def __init__(self, monitor="validation_loss", mode="min", period=1, verbose=False, **kw):
            super().__init__(); self.monitor=monitor; self.mode=mode; self.period=period

    if not hasattr(scvi_settings, "dl_pin_memory_gpu_training"):
        scvi_settings.dl_pin_memory_gpu_training = False
    scvi_model_utils.parse_use_gpu_arg = parse_use_gpu_arg
    scvi_callbacks.SaveBestState = SaveBestState

    class CompatTrainRunner(CurrentTrainRunner):
        def __init__(self, model, training_plan, data_splitter, max_epochs,
                     accelerator=None, devices=None, use_gpu=None, **kw):
            if accelerator is None:
                accelerator, pd = parse_use_gpu_arg(use_gpu=use_gpu, return_device=False)
                devices = pd if devices is None else devices
            if devices is None: devices = "auto"
            super().__init__(model=model, training_plan=training_plan, data_splitter=data_splitter,
                           max_epochs=max_epochs, accelerator=accelerator, devices=devices, **kw)
    def _run_training_core(self):
        import gc

        try:
            self.trainer.fit(self.training_plan, self.data_splitter, ckpt_path=self.ckpt_path)
        except BaseException as e:
            self._update_history()
            print("Exception raised during training.", type(e), e)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        self._update_history()

    CompatTrainRunner._run_training_core = _run_training_core

    # Wrap __init__ to capture ckpt_path from environment
    _orig_compat_init = CompatTrainRunner.__init__
    def _new_init(self, *a, **kw):
        _orig_compat_init(self, *a, **kw)
        self.ckpt_path = os.environ.get("_CPA_CKPT_PATH", None)
    CompatTrainRunner.__init__ = _new_init

    scvi_train.TrainRunner = CompatTrainRunner

    def patch_dataloader_splitter(cls):
        if getattr(cls, "_codex_loader_patch", False):
            return
        original_init = cls.__init__

        def __init__(self, *a, **kw):
            original_init(self, *a, **kw)
            self.data_loader_kwargs.update(getattr(cls, "_codex_data_loader_kwargs", {}))

        def train_dataloader(self):
            if len(self.train_idx) <= 0:
                return None
            from cpa._data import AnnDataLoader
            return AnnDataLoader(
                self.adata_manager, indices=self.train_idx, shuffle=True,
                pin_memory=self.pin_memory, **self.data_loader_kwargs)

        def val_dataloader(self):
            if len(self.val_idx) <= 0:
                return None
            from cpa._data import AnnDataLoader
            return AnnDataLoader(
                self.adata_manager, indices=self.val_idx, shuffle=False,
                pin_memory=self.pin_memory, **self.data_loader_kwargs)

        def test_dataloader(self):
            if len(self.test_idx) <= 0:
                return None
            from cpa._data import AnnDataLoader
            return AnnDataLoader(
                self.adata_manager, indices=self.test_idx, shuffle=False,
                pin_memory=self.pin_memory, **self.data_loader_kwargs)

        cls.__init__ = __init__
        cls.train_dataloader = train_dataloader
        cls.val_dataloader = val_dataloader
        cls.test_dataloader = test_dataloader
        cls._codex_loader_patch = True

    def patch_module_metrics(cls):
        if getattr(cls, "_codex_metric_patch", False):
            return

        def r2_metric(self, tensors, inference_outputs, generative_outputs, mode: str = "lfc"):
            from cpa._utils import CPA_REGISTRY_KEYS

            mode = mode.lower()
            assert mode in ["direct"]
            x = tensors[CPA_REGISTRY_KEYS.X_KEY]
            indices = tensors[CPA_REGISTRY_KEYS.CATEGORY_KEY].view(-1,)
            unique_indices = indices.unique()
            if len(unique_indices) == 0:
                return 0.0, 0.0

            r2_mean = 0.0
            r2_var = 0.0
            px = generative_outputs["px"]
            for ind in unique_indices:
                i_mask = indices == ind
                x_i = x[i_mask, :]
                if self.recon_loss == "gauss":
                    x_pred_mean = px.loc[i_mask, :]
                    x_pred_var = px.scale[i_mask, :] ** 2
                    if CPA_REGISTRY_KEYS.DEG_MASK_R2 in tensors.keys():
                        deg_mask = tensors[f"{CPA_REGISTRY_KEYS.DEG_MASK_R2}"][i_mask, :]
                        x_i *= deg_mask
                        x_pred_mean *= deg_mask
                        x_pred_var *= deg_mask
                    x_pred_mean = torch.nan_to_num(x_pred_mean, nan=0, posinf=1e3, neginf=-1e3)
                    x_pred_var = torch.nan_to_num(x_pred_var, nan=0, posinf=1e3, neginf=-1e3)
                    r2_mean += torch.nan_to_num(
                        self.metrics["r2_score"](x_pred_mean.mean(0), x_i.mean(0)),
                        nan=0.0).item()
                    r2_var += torch.nan_to_num(
                        self.metrics["r2_score"](x_pred_var.mean(0), x_i.var(0, unbiased=False)),
                        nan=0.0).item()
                elif self.recon_loss in ["nb", "zinb"]:
                    x_i = torch.log(1 + x_i)
                    x_pred = torch.log(1 + px.mu[i_mask, :])
                    x_pred = torch.nan_to_num(x_pred, nan=0, posinf=1e3, neginf=-1e3)
                    if CPA_REGISTRY_KEYS.DEG_MASK_R2 in tensors.keys():
                        deg_mask = tensors[f"{CPA_REGISTRY_KEYS.DEG_MASK_R2}"][i_mask, :]
                        x_i *= deg_mask
                        x_pred *= deg_mask
                    r2_mean += torch.nan_to_num(
                        self.metrics["r2_score"](x_pred.mean(0), x_i.mean(0)),
                        nan=0.0).item()
                    r2_var += torch.nan_to_num(
                        self.metrics["r2_score"](x_pred.var(0, unbiased=False), x_i.var(0, unbiased=False)),
                        nan=0.0).item()
            return r2_mean / len(unique_indices), r2_var / len(unique_indices)

        cls.r2_metric = r2_metric
        cls._codex_metric_patch = True

    def patch_epoch_metrics(cls):
        if getattr(cls, "_codex_epoch_patch", False):
            return

        def mean_nonzero(outputs, key):
            vals = [output[key] for output in outputs if output.get(key, 0.0) != 0.0]
            return float(np.mean(vals)) if vals else 0.0

        def training_epoch_end(self, outputs):
            for key in self.metrics:
                self.epoch_history[key].append(
                    0.0 if key in ["disnt_basal", "disnt_after"] else mean_nonzero(outputs, key))
            for covar, unique_covars in self.covars_encoder.items():
                if len(unique_covars) > 1:
                    for key in (f"adv_{covar}", f"penalty_{covar}", f"acc_{covar}"):
                        self.epoch_history[key].append(mean_nonzero(outputs, key))
            self.epoch_history["epoch"].append(self.current_epoch)
            self.epoch_history["mode"].append("train")
            self.log("recon", self.epoch_history["recon_loss"][-1], prog_bar=True)
            self.log("r2_mean", self.epoch_history["r2_mean"][-1], prog_bar=True)
            self.log("adv_loss", self.epoch_history["adv_loss"][-1], prog_bar=True)
            self.log("acc_pert", self.epoch_history["acc_perts"][-1], prog_bar=True)
            for covar, nc in self.covars_encoder.items():
                if len(nc) > 1:
                    self.log(f"acc_{covar}", self.epoch_history[f"acc_{covar}"][-1], prog_bar=True)
            if self.current_epoch > 1 and self.current_epoch % self.step_size_lr == 0:
                sch, sch_doser, sch_adv = self.lr_schedulers()
                sch.step(); sch_doser.step(); sch_adv.step()

        def validation_epoch_end(self, outputs):
            for key in self.metrics:
                self.epoch_history[key].append(mean_nonzero(outputs, key))
            for covar, unique_covars in self.covars_encoder.items():
                if len(unique_covars) > 1:
                    for key in (f"adv_{covar}", f"penalty_{covar}", f"acc_{covar}"):
                        self.epoch_history[key].append(mean_nonzero(outputs, key))
            self.epoch_history["epoch"].append(self.current_epoch)
            self.epoch_history["mode"].append("valid")
            cpa_metric = float(np.mean([output["cpa_metric"] for output in outputs])) if outputs else 0.0
            self.log("val_recon", self.epoch_history["recon_loss"][-1], prog_bar=True)
            self.log("cpa_metric", cpa_metric, prog_bar=False)
            self.log("disnt_basal", self.epoch_history["disnt_basal"][-1], prog_bar=True)
            self.log("disnt_after", self.epoch_history["disnt_after"][-1], prog_bar=True)
            self.log("val_r2_mean", self.epoch_history["r2_mean"][-1], prog_bar=True)
            self.log("val_r2_var", self.epoch_history["r2_var"][-1], prog_bar=False)
            self.log("val_KL", self.epoch_history["KL"][-1], prog_bar=True)
            if self.current_epoch % self.n_epochs_verbose == self.n_epochs_verbose - 1:
                print(f'\ndisnt_basal = {self.epoch_history["disnt_basal"][-1]}')
                print(f'disnt_after = {self.epoch_history["disnt_after"][-1]}')
                print(f'val_r2_mean = {self.epoch_history["r2_mean"][-1]}')
                print(f'val_r2_var = {self.epoch_history["r2_var"][-1]}')

        cls.training_epoch_end = training_epoch_end
        cls.validation_epoch_end = validation_epoch_end
        cls._codex_epoch_patch = True

    def patch_hooks(cls):
        if getattr(cls, "_codex_l2", False): return
        ots, ovs = cls.training_step, cls.validation_step
        otee, ovee = cls.training_epoch_end, cls.validation_epoch_end
        def ts(self, *a, **kw): o=ots(self,*a,**kw); self._cto.append(o); return o
        def vs(self, *a, **kw): o=ovs(self,*a,**kw); self._cvo.append(o); return o
        def otes(self): self._cto=[]
        def oves(self): self._cvo=[]
        def otee2(self):
            o=getattr(self,"_cto",[]);
            if o: otee(self,o)
            self._cto=[]
        def ovee2(self):
            o=getattr(self,"_cvo",[])
            if o: ovee(self,o)
            self._cvo=[]
        cls.training_step=ts; cls.validation_step=vs
        cls.on_train_epoch_start=otes; cls.on_validation_epoch_start=oves
        cls.on_train_epoch_end=otee2; cls.on_validation_epoch_end=ovee2
        delattr(cls,"training_epoch_end"); delattr(cls,"validation_epoch_end")
        cls._codex_l2 = True

    def patch_perturbation_network(cls):
        if getattr(cls, "_codex_shape_patch", False):
            return
        original_forward = cls.forward

        def normalize_combo_tensor(x):
            if not torch.is_tensor(x) or x.dim() <= 2:
                return x
            while x.dim() > 2:
                squeezed = False
                for dim in range(x.dim() - 1):
                    if x.shape[dim] == 1:
                        x = x.squeeze(dim)
                        squeezed = True
                        break
                if not squeezed:
                    x = x.reshape(-1, x.shape[-1])
            return x

        def forward(self, perts, dosages):
            original_shape = tuple(perts.shape) if torch.is_tensor(perts) else None
            perts = normalize_combo_tensor(perts)
            dosages = normalize_combo_tensor(dosages)
            if (
                original_shape is not None
                and original_shape != tuple(perts.shape)
                and not getattr(self, "_codex_shape_patch_reported", False)
            ):
                emit("pert_tensor_shape_normalized", original=original_shape, normalized=tuple(perts.shape))
                self._codex_shape_patch_reported = True
            return original_forward(self, perts, dosages)

        cls.forward = forward
        cls._codex_shape_patch = True

    def patch_module_batch_shapes(cls):
        if getattr(cls, "_codex_batch_shape_patch", False):
            return

        original_mixup_data = cls.mixup_data
        original_get_inference_input = cls._get_inference_input

        def unwrap_leading_batch_axis(self, tensors):
            from cpa._utils import CPA_REGISTRY_KEYS

            if not isinstance(tensors, dict):
                return tensors
            x = tensors.get(CPA_REGISTRY_KEYS.X_KEY)
            if not torch.is_tensor(x) or x.dim() <= 2 or x.shape[0] != 1:
                return tensors

            wrapped_batch_size = int(x.shape[1])
            changed = []
            for key, value in list(tensors.items()):
                if (
                    torch.is_tensor(value)
                    and value.dim() >= 2
                    and value.shape[0] == 1
                    and value.shape[1] == wrapped_batch_size
                ):
                    original_shape = tuple(value.shape)
                    tensors[key] = value.squeeze(0)
                    changed.append((str(key), original_shape, tuple(tensors[key].shape)))

            if changed and not getattr(self, "_codex_batch_shape_patch_reported", False):
                emit(
                    "batch_tensor_shape_normalized",
                    batch_size=wrapped_batch_size,
                    keys=[
                        {"key": key, "original": original, "normalized": normalized}
                        for key, original, normalized in changed
                    ],
                )
                self._codex_batch_shape_patch_reported = True
            return tensors

        def mixup_data(self, tensors, *a, **kw):
            return original_mixup_data(self, unwrap_leading_batch_axis(self, tensors), *a, **kw)

        def _get_inference_input(self, tensors):
            return original_get_inference_input(self, unwrap_leading_batch_axis(self, tensors))

        cls.mixup_data = mixup_data
        cls._get_inference_input = _get_inference_input
        cls._codex_batch_shape_patch = True

    def patch_cpa_load(cls):
        if getattr(cls, "_codex_load_patch", False):
            return

        def load_device_args(use_gpu):
            if use_gpu is False:
                return "cpu", "auto"
            if use_gpu is None or use_gpu == "auto" or use_gpu is True:
                return "auto", "auto"
            if isinstance(use_gpu, int):
                return "cuda", use_gpu
            if isinstance(use_gpu, str):
                value = use_gpu.lower()
                if value == "cpu":
                    return "cpu", "auto"
                if value in {"cuda", "gpu"}:
                    return "cuda", "auto"
                return "cuda", int(value)
            return "auto", "auto"

        @classmethod
        def load(cls_, dir_path, adata=None, use_gpu=None):
            with open(Path(dir_path) / "CPA_info.json") as f:
                total_dict = json.load(f)
            cls_.pert_encoder = total_dict["pert_encoder"]
            cls_.covars_encoder = total_dict["covars_encoder"]
            cls_.pert_smiles_map = total_dict.get("pert_smiles_map", None)

            accelerator, device = load_device_args(use_gpu)
            parent_load = cls_.__mro__[1].__dict__["load"].__func__
            model = parent_load(
                cls_,
                str(dir_path),
                adata=adata,
                accelerator=accelerator,
                device=device,
            )
            try:
                model.epoch_history = pd.read_csv(Path(dir_path) / "history.csv")
            except Exception:
                print("WARNING: The history was not found.")
            return model

        cls.load = load
        cls._codex_load_patch = True

    def patch_cpa_save(cls):
        if getattr(cls, "_codex_save_patch", False):
            return
        original_save = cls.save

        def save(self, dir_path, *a, **kw):
            if not is_primary_process():
                emit("rank_skip_model_save", rank=process_rank(), output=str(dir_path))
                return None
            return original_save(self, dir_path, *a, **kw)

        cls.save = save
        cls._codex_save_patch = True

    spec = importlib.util.find_spec("cpa")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Could not locate installed cpa package")
    pkg_dir = Path(next(iter(spec.submodule_search_locations)))
    pkg = types.ModuleType("cpa"); pkg.__path__ = [str(pkg_dir)]; sys.modules["cpa"] = pkg
    ms = importlib.util.spec_from_file_location("cpa._model", pkg_dir / "_model.py")
    mm = importlib.util.module_from_spec(ms); sys.modules["cpa._model"] = mm
    ms.loader.exec_module(mm)
    from cpa._task import CPATrainingPlan
    from cpa._module import CPAModule
    from cpa._data import AnnDataSplitter
    from cpa._utils import PerturbationNetwork
    patch_dataloader_splitter(AnnDataSplitter)
    patch_module_metrics(CPAModule)
    patch_module_batch_shapes(CPAModule)
    patch_epoch_metrics(CPATrainingPlan)
    patch_hooks(CPATrainingPlan)
    patch_perturbation_network(PerturbationNetwork)
    patch_cpa_load(mm.CPA)
    patch_cpa_save(mm.CPA)
    return mm.CPA


def reset_cpa(CPA):
    CPA.pert_encoder = None; CPA.covars_encoder = None; CPA.pert_smiles_map = None


# ==============================================================================
# scGPT Aligner (M1, M5)
# ==============================================================================

class Aligner(nn.Module):
    """MLP: scGPT 512d -> gene expression space."""
    def __init__(self, input_dim=512, output_dim=18211, hidden_dims=(1024, 2048, 4096, 8192), dropout=0.1):
        super().__init__()
        layers = []; prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def train_aligner(adata, scgpt_key="X_scGPT", device="cuda", epochs=30, lr=1e-3, batch_size=512):
    """Train MLP aligner: scGPT embeddings -> gene expression (on control cells)."""
    ctrl_mask = adata.obs["neg_control"].astype(int).eq(1).to_numpy()
    X_scgpt = adata.obsm[scgpt_key][ctrl_mask].astype(np.float32)
    X_expr = dense(adata.X[ctrl_mask]) if not isinstance(adata.X, np.ndarray) else adata.X[ctrl_mask]

    # Use counts layer if available
    if "counts" in adata.layers:
        X_expr = dense(adata.layers["counts"][ctrl_mask])

    X_expr = np.log1p(X_expr)  # log1p transform for alignment

    n_genes = X_expr.shape[1]
    aligner = Aligner(input_dim=X_scgpt.shape[1], output_dim=n_genes).to(device)
    optimizer = torch.optim.Adam(aligner.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = X_scgpt.shape[0]
    idx = np.arange(n)

    for epoch in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0; nb = 0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start+batch_size]
            x = torch.tensor(X_scgpt[batch_idx], dtype=torch.float32).to(device)
            y = torch.tensor(X_expr[batch_idx], dtype=torch.float32).to(device)
            pred = aligner(x)
            loss = criterion(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f"  Aligner epoch {epoch+1}/{epochs}: loss={total_loss/nb:.4f}")

    return aligner


def build_aligned_adata(adata, aligner, scgpt_key, device):
    """Project scGPT through aligner -> new AnnData with aligned X."""
    aligner.eval()
    X_scgpt = adata.obsm[scgpt_key].astype(np.float32)
    n = X_scgpt.shape[0]
    aligned = np.zeros((n, adata.n_vars), dtype=np.float32)
    bs = 512
    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch = torch.tensor(X_scgpt[start:end], dtype=torch.float32).to(device)
            aligned[start:end] = aligner(batch).cpu().numpy()
    aligned = np.clip(aligned, 0, None)

    new_adata = ad.AnnData(X=aligned, obs=adata.obs.copy(), var=adata.var.copy(), uns=adata.uns.copy())
    new_adata.layers["counts"] = aligned.copy()
    new_adata.obsm = {k: v.copy() for k, v in adata.obsm.items()}
    return new_adata


# ==============================================================================
# MolFormer drug embeddings (M4, M5)
# ==============================================================================

def build_molformer_embeddings(parquet_path, pert_encoder, adata):
    """Build frozen MolFormer embedding aligned to CPA perturbation encoder."""
    drug_df = pd.read_parquet(parquet_path)
    mdim = drug_df.shape[1]
    s2e = {smi: drug_df.loc[smi].values.astype(np.float32) for smi in drug_df.index}

    # Build drug_name -> SMILES mapping from adata
    drug_smiles = {}
    if "SMILES" in adata.obs.columns and "condition" in adata.obs.columns:
        pairs = adata.obs[["condition", "SMILES"]].dropna().drop_duplicates()
        for cond, smi in pairs.itertuples(index=False):
            cond, smi = str(cond), str(smi)
            if cond and cond != "control" and smi:
                drug_smiles.setdefault(cond, smi)

    from cpa._utils import CPA_REGISTRY_KEYS
    n_perts = len(pert_encoder)
    emat = np.zeros((n_perts, mdim), dtype=np.float32)

    from rdkit import Chem
    matched = 0
    for dname, idx in pert_encoder.items():
        if dname in ("<PAD>", "control"): continue
        smi = drug_smiles.get(dname, "")
        if smi in s2e:
            emat[idx] = s2e[smi]; matched += 1; continue
        try:
            canonical = Chem.CanonSmiles(smi) if smi else ""
        except Exception:
            canonical = ""
        if canonical and canonical in s2e:
            emat[idx] = s2e[canonical]; matched += 1

    emb = nn.Embedding(n_perts, mdim, padding_idx=CPA_REGISTRY_KEYS.PADDING_IDX)
    emb.weight.data.copy_(torch.tensor(emat))
    emb.weight.requires_grad = False
    emit("molformer_built", n_perts=n_perts, dim=mdim, matched=matched)
    return emb


# ==============================================================================
# CPA training
# ==============================================================================

def train_cpa_variant(CPA, adata, variant, args):
    """Train a CPA variant and return the model."""
    reset_cpa(CPA)

    is_scgpt = variant in ("M1", "M5")
    is_molformer = variant in ("M4", "M5")

    # Setup AnnData
    setup_kwargs = {
        "perturbation_key": "condition",
        "control_group": "control",
        "dosage_key": "dose_val",
        "categorical_covariate_keys": ["cell_type"],
        "layer": "counts",
        "is_count_data": not is_scgpt,  # Gauss loss for scGPT variants
    }
    CPA.setup_anndata(adata, **setup_kwargs)

    # Build MolFormer embeddings if needed
    drug_emb = None
    if is_molformer:
        drug_emb = build_molformer_embeddings(args.molformer_parquet, CPA.pert_encoder, adata)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    use_gpu = False if args.device == "cpu" else ("auto" if args.device == "auto" else True)
    recon_loss = "gauss" if is_scgpt else "nb"

    model_kwargs = dict(
        split_key=args.split_key,
        train_split="train",
        valid_split="test",
        test_split="ood",
        n_latent=args.n_latent,
        recon_loss=recon_loss,
        seed=args.seed,
    )
    if drug_emb is not None:
        model_kwargs["drug_embeddings"] = drug_emb

    model = CPA(adata, **model_kwargs)

    model_dir = DELIVERY / "models" / f"CPA_NIPS_{variant}_model"
    # Only back up model_dir on fresh runs (not resume)
    if is_primary_process() and model_dir.exists() and args.fresh:
        backup = model_dir.with_name(f"{model_dir.name}.previous")
        if backup.exists():
            stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            backup = model_dir.with_name(f"{model_dir.name}.previous.{stamp}")
        model_dir.rename(backup)

    # Find checkpoint for resume
    ckpt_path = None
    if not args.fresh:
        ckpt_path = find_last_checkpoint(model_dir)
        if ckpt_path:
            emit("resume_from_checkpoint", variant=variant, checkpoint=ckpt_path)
            print(f"  Resuming from checkpoint: {ckpt_path}")
            os.environ["_CPA_CKPT_PATH"] = ckpt_path
        else:
            print(f"  No checkpoint found, training from scratch.")
            os.environ.pop("_CPA_CKPT_PATH", None)
    else:
        os.environ.pop("_CPA_CKPT_PATH", None)

    train_kwargs = lightning_train_kwargs(args)
    tb_log_dir = model_dir / "tensorboard_logs"
    tb_logger = TensorBoardLogger(save_dir=str(tb_log_dir), name=f"cpa_{variant}")
    emit("train_start", variant=variant, recon_loss=recon_loss, has_molformer=is_molformer,
         accelerator=train_kwargs.get("accelerator"), devices=train_kwargs.get("devices"),
         strategy=train_kwargs.get("strategy", "auto"), num_workers=args.num_workers,
         tensorboard_log_dir=str(tb_logger.log_dir), resume=ckpt_path is not None)
    previous_active_variant = os.environ.get("CPA_NIPS_ACTIVE_VARIANT")
    os.environ["CPA_NIPS_ACTIVE_VARIANT"] = variant
    try:
        model.train(
            max_epochs=args.max_epochs,
            use_gpu=use_gpu,
            batch_size=args.batch_size,
            save_path=str(model_dir),
            check_val_every_n_epoch=1,
            early_stopping_patience=args.early_stopping_patience,
            plan_kwargs={"do_clip_grad": True, "gradient_clip_value": 3.0, "n_epochs_verbose": 1},
            log_every_n_steps=25,
            enable_progress_bar=True,
            logger=tb_logger,
            **train_kwargs,
        )
    finally:
        os.environ.pop("_CPA_CKPT_PATH", None)
        if previous_active_variant is None:
            os.environ.pop("CPA_NIPS_ACTIVE_VARIANT", None)
        else:
            os.environ["CPA_NIPS_ACTIVE_VARIANT"] = previous_active_variant

    history = model.epoch_history.copy()
    if is_primary_process():
        model_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(model_dir / "history.csv", index=False)
        emit("train_complete", variant=variant, epochs=int(history["epoch"].nunique()))
    mark_done(variant, "training_complete")
    return model


# ==============================================================================
# Prediction
# ==============================================================================

def predict_ood_groups(model, adata, variant, args):
    """Predict for cells requested by --eval-settings."""
    from cpa._utils import CPA_REGISTRY_KEYS

    split_values = adata.obs[args.split_key].astype(str)
    eval_settings = [str(s) for s in args.eval_settings]
    if "all" in eval_settings:
        pred_mask = np.ones(adata.n_obs, dtype=bool)
    else:
        pred_mask = split_values.isin(eval_settings).to_numpy()
    if not pred_mask.any():
        pred_mask = (split_values == "ood").to_numpy()
        emit("prediction_mask_fallback", requested=eval_settings, fallback="ood")
    pred_adata = adata[pred_mask].copy()
    emit(
        "prediction_cells_selected",
        variant=variant,
        requested=eval_settings,
        splits=pred_adata.obs[args.split_key].astype(str).value_counts().to_dict(),
    )

    max_comb_len = int(CPA_REGISTRY_KEYS.MAX_COMB_LENGTH)
    pert_embs, dose_embs = [], []
    for _, row in pred_adata.obs.iterrows():
        drug = str(row["condition"])
        dose = float(row.get("dose_val", 1.0))
        if drug in model.pert_encoder:
            perts = [model.pert_encoder[drug]] + [CPA_REGISTRY_KEYS.PADDING_IDX] * (max_comb_len - 1)
        else:
            perts = [CPA_REGISTRY_KEYS.PADDING_IDX] * max_comb_len
        pert_embs.append(perts)
        dose_embs.append([dose] + [0.0] * (max_comb_len - 1))

    pred_adata.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS] = np.array(pert_embs, dtype=np.int64)
    pred_adata.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS_DOSAGES] = np.array(dose_embs, dtype=np.float32)

    cat_key = CPA_REGISTRY_KEYS.CATEGORY_KEY
    if cat_key in pred_adata.obs:
        pred_adata.obs[cat_key] = pred_adata.obs[["cell_type", "condition"]].apply(
            lambda r: "_".join(r.astype(str)), axis=1)

    model.predict(pred_adata, batch_size=args.batch_size, n_samples=1, return_mean=True)
    pred_x = np.asarray(pred_adata.obsm["CPA_pred"], dtype=np.float32)
    pred_x = np.nan_to_num(pred_x, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    pred_x[pred_x < 0] = 0.0

    pred_path = DELIVERY / "predictions" / f"CPA_NIPS_{variant}_pred.h5ad"
    pred = ad.AnnData(X=pred_x, obs=pred_adata.obs.copy(), var=adata.var.copy(),
                      uns={"prediction": {"method": f"CPA_{variant}", "dataset": "NIPS"}})
    pred.obs_names = pred_adata.obs_names.copy()
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(pred_path, compression="gzip")
    emit("prediction_saved", variant=variant, output=str(pred_path), shape=list(pred.shape))
    return pred


# ==============================================================================
# NIPS protocol evaluation
# ==============================================================================

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return max(1 - ss_res / ss_tot, 0.0) if ss_tot > 0 else 0.0


def sinkhorn_distance(x_pred, x_true, samples=512, seed=7):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        configure_keops_cuda()

    from geomloss import SamplesLoss

    rng = np.random.default_rng(seed)
    if x_pred.shape[0] > samples:
        x_pred = x_pred[rng.choice(x_pred.shape[0], samples, replace=False)]
    if x_true.shape[0] > samples:
        x_true = x_true[rng.choice(x_true.shape[0], samples, replace=False)]
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if not getattr(sinkhorn_distance, "_device_reported", False):
        emit("sinkhorn_device", device=str(device))
        sinkhorn_distance._device_reported = True
    loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized")
    with torch.no_grad():
        return float(loss(torch.as_tensor(x_pred, dtype=torch.float32, device=device),
                          torch.as_tensor(x_true, dtype=torch.float32, device=device)).item())


def _safe_pearson(a, b):
    r = stats.pearsonr(a, b).statistic
    return max(r, 0.0) if not np.isnan(r) else 0.0


def calc_metrics_group(yt_m, yp_m, ctrl_m, y_true, y_pred, idx_de):
    yt_de = yt_m[idx_de].copy(); yp_de = yp_m[idx_de].copy()
    if yt_de.sum() == 0: yt_de[0] += 1e-6
    if yp_de.sum() == 0: yp_de[0] += 1e-6

    m = {}
    m["r2score"] = compute_r2(yt_m, yp_m)
    m["r2score_de"] = compute_r2(yt_m[idx_de], yp_m[idx_de])
    m["pearson"] = _safe_pearson(yt_m, yp_m)
    m["pearson_de"] = _safe_pearson(yt_de, yp_de)
    m["mse"] = float(mse_sklearn(yt_m, yp_m))
    m["mse_de"] = float(mse_sklearn(yt_m[idx_de], yp_m[idx_de]))
    m["pearson_delta"] = _safe_pearson(yt_m - ctrl_m, yp_m - ctrl_m)
    m["pearson_delta_de"] = _safe_pearson(yt_de - ctrl_m[idx_de], yp_de - ctrl_m[idx_de])

    if y_pred[:, idx_de].sum() == 0 and y_true[:, idx_de].sum() == 0:
        m["sinkhorn_de"] = 0.0
    else:
        m["sinkhorn_de"] = sinkhorn_distance(y_pred[:, idx_de], y_true[:, idx_de])
    return m


def evaluate_split(adata, pred, split_key, setting, args):
    if setting == "ood":
        split_mask = adata.obs[split_key].astype(str) == "ood"
    elif setting == "test":
        split_mask = adata.obs[split_key].astype(str) == "test"
    else:
        split_mask = np.ones(adata.n_obs, dtype=bool)

    obs_split = adata.obs.loc[split_mask]
    groups = obs_split["cov_drug_name"].unique()
    deg_dict = adata.uns["rank_genes_groups_cov"]
    var_names = adata.var_names
    var_pos = {gene: i for i, gene in enumerate(var_names)}
    deg_idx_cache: dict[str, np.ndarray] = {}
    neg_ctrl = pd.to_numeric(adata.obs["neg_control"], errors="coerce").fillna(0).astype(int).eq(1).to_numpy()

    eval_scores = {}
    skipped = {"treated_too_few": 0, "dmso_control": 0, "deg_missing": 0,
               "deg_too_few": 0, "ctrl_too_few": 0, "no_pred": 0}

    for group in groups:
        gs = str(group)
        ct = gs.split("_")[0]
        drug = "_".join(gs.split("_")[1:])

        if "dmso" in gs.lower() or "control" in gs.lower():
            skipped["dmso_control"] += 1; continue

        treated_mask = (split_mask.to_numpy() &
                       (adata.obs["cov_drug_name"].astype(str) == gs).to_numpy() & ~neg_ctrl)
        if int(treated_mask.sum()) <= args.min_treated:
            skipped["treated_too_few"] += 1; continue

        ctrl_mask = (split_mask.to_numpy() &
                    (adata.obs["cell_type"].astype(str) == ct).to_numpy() & neg_ctrl)
        if ctrl_mask.sum() < args.min_ctrl:
            ctrl_mask = (adata.obs["cell_type"].astype(str) == ct).to_numpy() & neg_ctrl
        if int(ctrl_mask.sum()) < args.min_ctrl:
            skipped["ctrl_too_few"] += 1; continue

        pred_mask = ((pred.obs["cell_type"].astype(str) == ct).to_numpy() &
                    (pred.obs["condition"].astype(str) == drug).to_numpy())
        if args.split_key in pred.obs.columns and setting in {"ood", "test", "train"}:
            pred_mask &= (pred.obs[args.split_key].astype(str) == setting).to_numpy()
        if pred_mask.sum() == 0:
            skipped["no_pred"] += 1; continue

        deg_key = gs if gs in deg_dict else gs.replace("_", "|", 1)
        if deg_key not in deg_dict:
            skipped["deg_missing"] += 1; continue
        if deg_key not in deg_idx_cache:
            deg_genes = set(deg_dict[deg_key])
            deg_idx_cache[deg_key] = np.array(
                [var_pos[g] for g in deg_genes if g in var_pos],
                dtype=np.int64,
            )
        idx_de = deg_idx_cache[deg_key]
        if len(idx_de) < args.min_deg:
            skipped["deg_too_few"] += 1; continue

        Y_true = dense(adata.X[treated_mask])
        Y_ctrl = dense(adata.X[ctrl_mask])
        Y_pred = dense(pred.X[pred_mask])

        y, p, c = Y_true.mean(axis=0), Y_pred.mean(axis=0), Y_ctrl.mean(axis=0)
        m = calc_metrics_group(y, p, c, Y_true, Y_pred, idx_de)
        m["n_treated"], m["n_ctrl"], m["n_pred"] = int(treated_mask.sum()), int(ctrl_mask.sum()), int(pred_mask.sum())
        eval_scores[gs] = m

    if not eval_scores:
        return None, skipped

    metric_names = ["r2score", "r2score_de", "pearson", "pearson_de", "mse", "mse_de",
                    "pearson_delta", "pearson_delta_de", "sinkhorn_de"]
    macro_avg = {k: float(np.mean([eval_scores[g][k] for g in eval_scores])) for k in metric_names}
    macro_avg["n_valid_groups"] = len(eval_scores)
    macro_avg["skipped"] = skipped
    return macro_avg, skipped


# ==============================================================================
# Main
# ==============================================================================

def run_variant(variant, CPA, adata, args):
    """Run a single CPA variant: train -> predict -> evaluate."""
    print(f"\n{'='*70}")
    print(f"  CPA Variant: {variant}")
    if variant == "M0": print("  Cell: raw counts | Drug: learnable | Loss: NB")
    elif variant == "M1": print("  Cell: scGPT aligned | Drug: learnable | Loss: Gauss")
    elif variant == "M4": print("  Cell: raw counts | Drug: MolFormer 768d | Loss: NB")
    elif variant == "M5": print("  Cell: scGPT aligned | Drug: MolFormer 768d | Loss: Gauss")
    if not args.fresh:
        prog = load_progress(variant)
        done_stages = [k for k, v in prog.items() if v]
        if done_stages:
            print(f"  Resume: already completed stages: {done_stages}")
    print(f"{'='*70}")

    # For scGPT variants, prepare aligned adata (with caching)
    run_adata = adata
    if variant in ("M1", "M5"):
        aligned_cache = DELIVERY / "models" / f"CPA_NIPS_{variant}_aligned.h5ad"
        aligner_file = _aligner_path(variant)

        if not args.fresh and aligned_cache.exists():
            print(f"\nLoading cached aligned AnnData: {aligned_cache}")
            run_adata = ad.read_h5ad(str(aligned_cache))
            print(f"  Aligned shape: {run_adata.shape}")
        elif not args.fresh and is_done(variant, "aligner_trained") and aligner_file.exists():
            print(f"\nLoading saved aligner: {aligner_file}")
            device = torch.device(training_torch_device(args))
            n_genes = adata.n_vars
            aligner = Aligner(input_dim=adata.obsm["X_scGPT"].shape[1], output_dim=n_genes).to(device)
            aligner.load_state_dict(torch.load(aligner_file, map_location=device))
            print("Building aligned AnnData...")
            run_adata = build_aligned_adata(adata, aligner, "X_scGPT", str(device))
            print(f"  Aligned shape: {run_adata.shape}")
            if is_primary_process():
                run_adata.write_h5ad(str(aligned_cache), compression="gzip")
                print(f"  Cached aligned AnnData: {aligned_cache}")
            mark_done(variant, "aligned_built")
        else:
            print("\nTraining scGPT aligner (512d -> gene space)...")
            device = torch.device(training_torch_device(args))
            aligner = train_aligner(adata, scgpt_key="X_scGPT", device=str(device), epochs=30)
            if is_primary_process():
                torch.save(aligner.state_dict(), aligner_file)
                print(f"  Aligner saved: {aligner_file}")
            mark_done(variant, "aligner_trained")
            print("Building aligned AnnData...")
            run_adata = build_aligned_adata(adata, aligner, "X_scGPT", str(device))
            print(f"  Aligned shape: {run_adata.shape}")
            if is_primary_process():
                run_adata.write_h5ad(str(aligned_cache), compression="gzip")
                print(f"  Cached aligned AnnData: {aligned_cache}")
            mark_done(variant, "aligned_built")

    # Train
    pred_path = DELIVERY / "predictions" / f"CPA_NIPS_{variant}_pred.h5ad"
    pred = None

    skip_train = args.skip_train or (not args.fresh and is_done(variant, "training_complete"))
    if skip_train:
        print(f"\nSkipping training (already completed).")
        model = None
    else:
        model = train_cpa_variant(CPA, run_adata, variant, args)

    if not is_primary_process():
        emit("rank_skip_posttrain", variant=variant, rank=process_rank())
        return True

    # Predict (skip if already saved)
    skip_pred = not args.fresh and pred_path.exists()
    if pred is None:
        if skip_pred:
            print(f"\nLoading existing prediction: {pred_path}")
            pred = ad.read_h5ad(str(pred_path))
        else:
            if model is None:
                print("Loading existing model for prediction...")
                from cpa._model import CPA as CPAModel
                model_dir = DELIVERY / "models" / f"CPA_NIPS_{variant}_model"
                model = CPAModel.load(model_dir, run_adata)
            pred = predict_ood_groups(model, run_adata, variant, args)
    mark_done(variant, "prediction_saved")

    # Evaluate (skip if already done)
    out_dir = DELIVERY / "evaluation" / "nips_dataset"
    out_json = out_dir / f"CPA_{variant}_{args.split_key}_metrics.json"
    skip_eval = args.skip_eval or (not args.fresh and is_done(variant, "eval_done") and out_json.exists())

    if skip_eval:
        print(f"\nSkipping evaluation (already completed).")
        if out_json.exists():
            print(f"  Results: {out_json}")
    else:
        all_results = {}
        for setting in args.eval_settings:
            print(f"\n  Evaluating: {setting}")
            macro_avg, skipped_groups = evaluate_split(run_adata, pred, args.split_key, setting, args)
            if macro_avg is None:
                print(f"    No valid groups. Skipped: {skipped_groups}")
                all_results[setting] = {"status": "no_valid_groups", "skipped": skipped_groups}
                continue
            all_results[setting] = {"macro_avg": macro_avg}
            print(f"    Valid groups: {macro_avg['n_valid_groups']}")
            for k in ["r2score","r2score_de","pearson","pearson_de","mse","mse_de",
                      "pearson_delta","pearson_delta_de","sinkhorn_de"]:
                print(f"    {k:<25} {macro_avg[k]:>10.4f}")
            n_no_pred = skipped_groups.get("no_pred", 0)
            if n_no_pred > 0:
                print(f"    WARNING: CPA could not predict {n_no_pred} groups (unseen cell types)")

        class NpEnc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.floating, np.integer)): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return super().default(o)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(all_results, indent=2, cls=NpEnc))
        print(f"\n  Results: {out_json}")
        mark_done(variant, "eval_done")

    return True


def main():
    args = parse_args()
    log_path = DELIVERY / "logs" / "CPA_NIPS_dataset_training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as lh, \
         contextlib.redirect_stdout(Tee(sys.stdout, lh)), \
         contextlib.redirect_stderr(Tee(sys.stderr, lh)):

        emit("run_start", variant=args.variant, fresh=args.fresh)
        validate_cuda_devices(args)

        # Determine variants to run
        active_variant = os.environ.get("CPA_NIPS_ACTIVE_VARIANT")
        if not is_primary_process() and active_variant:
            variants = [active_variant]
            emit("ddp_child_variant_selected", variant=active_variant, rank=process_rank())
        else:
            variants = ["M0", "M1", "M4", "M5"] if args.variant == "all" else [args.variant]

        if args.fresh:
            clear_progress(variants)

        CPA = install_cpa_compat()
        configure_cpa_dataloading(args)

        print(f"Loading NIPS dataset: {args.adata}")
        adata = ad.read_h5ad(args.adata)
        adata.obs = adata.obs.copy()
        if "control" not in adata.obs.columns:
            adata.obs["control"] = adata.obs["neg_control"].astype(int).eq(1)

        required = {"condition", "dose_val", "cell_type", "neg_control", "cov_drug_name", args.split_key}
        missing = required - set(adata.obs.columns)
        if missing:
            raise ValueError(f"Missing obs columns: {missing}")
        print(f"  Shape: {adata.shape}, splits: {adata.obs[args.split_key].value_counts().to_dict()}")
        print(f"  DEGs: {len(adata.uns.get('rank_genes_groups_cov', {}))} groups")

        for v in variants:
            run_variant(v, CPA, adata, args)
            if not is_primary_process():
                return

        emit("run_complete")


if __name__ == "__main__":
    main()
