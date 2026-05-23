#!/usr/bin/env python
"""Run CPA (M0/M1/M4/M5) on the GBM NIPS-Ready dataset.

GBM dataset: 169,972 cells x 5,000 genes
  - Patients: 21 (PW029-PW040, GS359-GS789)
  - Drugs: 7 (Ana-12, Etoposide, Ispenisib, Panobinostat, RO4929097, Tazemetostat, Temozolomide)
  - OOD: PW034_Panobinostat (unseen patient-drug combination)
  - DEGs: 18 groups, 50 genes each

Variants:
  M0: raw counts, learnable drug embedding, NB loss
  M1: scGPT 512d -> Aligner MLP -> 5000d gene space, learnable drug embedding, Gauss loss
  M4: raw counts, MolFormer 768d frozen drug embedding, NB loss
  M5: scGPT -> Aligner -> gene space, MolFormer frozen drug embedding, Gauss loss

Usage:
  conda activate plknature
  python run_cpa_gbm_dataset.py --variant all
  python run_cpa_gbm_dataset.py --variant M1
  python run_cpa_gbm_dataset.py --variant M5 --fresh
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
    p = argparse.ArgumentParser(description="CPA variants on GBM NIPS-Ready dataset")
    p.add_argument("--adata", type=Path, default=DELIVERY / "dataset" / "GBM_NIPS_Ready.h5ad")
    p.add_argument("--molformer-parquet", type=Path,
                   default=ROOT / "01_REUSABLE_ASSETS" / "embeddings" / "GBM_molformer_drug_emb.parquet")
    p.add_argument("--molformer-metadata", type=Path,
                   default=ROOT / "01_REUSABLE_ASSETS" / "embeddings" / "GBM_molformer_drug_emb.metadata.json")
    p.add_argument("--variant", default="all", choices=["M0", "M1", "M4", "M5", "all"],
                   help="CPA variant to run, or 'all' for sequential")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-latent", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--fresh", action="store_true",
                   help="Force fresh run: clear progress, ignore existing checkpoints and cached data.")
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
    return DELIVERY / "models" / f"CPA_GBM_{variant}_progress.json"


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
    for v in variants:
        p = _progress_path(variant=v)
        if p.exists():
            p.unlink()
            emit("progress_cleared", variant=v)
        aligned_cache = DELIVERY / "models" / f"CPA_GBM_{v}_aligned.h5ad"
        if aligned_cache.exists():
            aligned_cache.unlink()
            emit("aligned_cache_cleared", variant=v)


def is_done(variant: str, stage: str) -> bool:
    return load_progress(variant).get(stage, False)


def find_last_checkpoint(model_dir: Path) -> str | None:
    ckpt_dir = model_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return str(ckpts[-1])
    last = model_dir / "last.ckpt"
    if last.exists():
        return str(last)
    ckpts = sorted(model_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        return str(ckpts[-1])
    return None


def _aligner_path(variant: str) -> Path:
    return DELIVERY / "models" / f"CPA_GBM_{variant}_aligner.pt"


# ==============================================================================
# CPA compatibility patches (shared with NIPS script)
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
            if len(self.train_idx) <= 0: return None
            from cpa._data import AnnDataLoader
            return AnnDataLoader(self.adata_manager, indices=self.train_idx, shuffle=True,
                                pin_memory=self.pin_memory, **self.data_loader_kwargs)
        def val_dataloader(self):
            if len(self.val_idx) <= 0: return None
            from cpa._data import AnnDataLoader
            return AnnDataLoader(self.adata_manager, indices=self.val_idx, shuffle=False,
                                pin_memory=self.pin_memory, **self.data_loader_kwargs)
        def test_dataloader(self):
            if len(self.test_idx) <= 0: return None
            from cpa._data import AnnDataLoader
            return AnnDataLoader(self.adata_manager, indices=self.test_idx, shuffle=False,
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
            if len(unique_indices) == 0: return 0.0, 0.0
            r2_mean = 0.0; r2_var = 0.0
            px = generative_outputs["px"]
            for ind in unique_indices:
                i_mask = indices == ind
                x_i = x[i_mask, :]
                if self.recon_loss == "gauss":
                    x_pred_mean = px.loc[i_mask, :]
                    x_pred_var = px.scale[i_mask, :] ** 2
                    if CPA_REGISTRY_KEYS.DEG_MASK_R2 in tensors.keys():
                        deg_mask = tensors[f"{CPA_REGISTRY_KEYS.DEG_MASK_R2}"][i_mask, :]
                        x_i *= deg_mask; x_pred_mean *= deg_mask; x_pred_var *= deg_mask
                    x_pred_mean = torch.nan_to_num(x_pred_mean, nan=0, posinf=1e3, neginf=-1e3)
                    x_pred_var = torch.nan_to_num(x_pred_var, nan=0, posinf=1e3, neginf=-1e3)
                    r2_mean += torch.nan_to_num(
                        self.metrics["r2_score"](x_pred_mean.mean(0), x_i.mean(0)), nan=0.0).item()
                    r2_var += torch.nan_to_num(
                        self.metrics["r2_score"](x_pred_var.mean(0), x_i.var(0, unbiased=False)), nan=0.0).item()
                elif self.recon_loss in ["nb", "zinb"]:
                    x_i = torch.log(1 + x_i)
                    x_pred = torch.log(1 + px.mu[i_mask, :])
                    x_pred = torch.nan_to_num(x_pred, nan=0, posinf=1e3, neginf=-1e3)
                    if CPA_REGISTRY_KEYS.DEG_MASK_R2 in tensors.keys():
                        deg_mask = tensors[f"{CPA_REGISTRY_KEYS.DEG_MASK_R2}"][i_mask, :]
                        x_i *= deg_mask; x_pred *= deg_mask
                    r2_mean += torch.nan_to_num(
                        self.metrics["r2_score"](x_pred.mean(0), x_i.mean(0)), nan=0.0).item()
                    r2_var += torch.nan_to_num(
                        self.metrics["r2_score"](x_pred.var(0, unbiased=False), x_i.var(0, unbiased=False)), nan=0.0).item()
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
                        x = x.squeeze(dim); squeezed = True; break
                if not squeezed:
                    x = x.reshape(-1, x.shape[-1])
            return x
        def forward(self, perts, dosages):
            original_shape = tuple(perts.shape) if torch.is_tensor(perts) else None
            perts = normalize_combo_tensor(perts)
            dosages = normalize_combo_tensor(dosages)
            if (original_shape is not None and original_shape != tuple(perts.shape)
                    and not getattr(self, "_codex_shape_patch_reported", False)):
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
            if not isinstance(tensors, dict): return tensors
            x = tensors.get(CPA_REGISTRY_KEYS.X_KEY)
            if not torch.is_tensor(x) or x.dim() <= 2 or x.shape[0] != 1: return tensors
            wrapped_batch_size = int(x.shape[1])
            changed = []
            for key, value in list(tensors.items()):
                if (torch.is_tensor(value) and value.dim() >= 2
                        and value.shape[0] == 1 and value.shape[1] == wrapped_batch_size):
                    original_shape = tuple(value.shape)
                    tensors[key] = value.squeeze(0)
                    changed.append((str(key), original_shape, tuple(tensors[key].shape)))
            if changed and not getattr(self, "_codex_batch_shape_patch_reported", False):
                emit("batch_tensor_shape_normalized", batch_size=wrapped_batch_size,
                     keys=[{"key": k, "original": o, "normalized": n} for k, o, n in changed])
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
            if use_gpu is False: return "cpu", "auto"
            if use_gpu is None or use_gpu == "auto" or use_gpu is True: return "auto", "auto"
            if isinstance(use_gpu, int): return "cuda", use_gpu
            if isinstance(use_gpu, str):
                value = use_gpu.lower()
                if value == "cpu": return "cpu", "auto"
                if value in {"cuda", "gpu"}: return "cuda", "auto"
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
            model = parent_load(cls_, str(dir_path), adata=adata, accelerator=accelerator, device=device)
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
    """MLP: scGPT 512d -> gene expression space (5000d for GBM)."""
    def __init__(self, input_dim=512, output_dim=5000, hidden_dims=(1024, 2048, 4096), dropout=0.1):
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
    ctrl = control_mask(adata.obs)
    X_scgpt = adata.obsm[scgpt_key][ctrl].astype(np.float32)
    X_expr = dense(adata.layers["counts"][ctrl])
    X_expr = np.log1p(X_expr)

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

def build_molformer_embeddings(parquet_path, metadata_path, pert_encoder, adata):
    """Build frozen Molformer embedding aligned to CPA perturbation encoder."""
    drug_df = pd.read_parquet(parquet_path)
    mdim = drug_df.shape[1]
    s2e = {smi: drug_df.loc[smi].values.astype(np.float32) for smi in drug_df.index}

    # Build drug_name -> SMILES mapping from metadata JSON
    drug_smiles = {}
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        for dname, info in meta.items():
            csmi = info.get("canonical_smiles", "")
            if csmi:
                drug_smiles[dname] = csmi
    else:
        # Fallback: use adata obs
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

    setup_kwargs = {
        "perturbation_key": "condition",
        "control_group": "control",
        "dosage_key": "dose_val",
        "categorical_covariate_keys": ["cell_type"],
        "layer": "counts",
        "is_count_data": not is_scgpt,
    }
    CPA.setup_anndata(adata, **setup_kwargs)

    drug_emb = None
    if is_molformer:
        drug_emb = build_molformer_embeddings(
            args.molformer_parquet, args.molformer_metadata, CPA.pert_encoder, adata)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    use_gpu = False if args.device == "cpu" else ("auto" if args.device == "auto" else True)
    recon_loss = "gauss" if is_scgpt else "nb"

    model_kwargs = dict(
        split_key="split",
        train_split="train",
        valid_split="valid",
        test_split="ood",
        n_latent=args.n_latent,
        recon_loss=recon_loss,
        seed=args.seed,
    )
    if drug_emb is not None:
        model_kwargs["drug_embeddings"] = drug_emb

    model = CPA(adata, **model_kwargs)

    model_dir = DELIVERY / "models" / f"CPA_GBM_{variant}_model"
    if model_dir.exists() and args.fresh:
        backup = model_dir.with_name(f"{model_dir.name}.previous")
        if backup.exists():
            stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            backup = model_dir.with_name(f"{model_dir.name}.previous.{stamp}")
        model_dir.rename(backup)

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

    tb_log_dir = model_dir / "tensorboard_logs"
    tb_logger = TensorBoardLogger(save_dir=str(tb_log_dir), name=f"cpa_gbm_{variant}")
    emit("train_start", variant=variant, recon_loss=recon_loss, has_molformer=is_molformer,
         tensorboard_log_dir=str(tb_logger.log_dir), resume=ckpt_path is not None)
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
        )
    finally:
        os.environ.pop("_CPA_CKPT_PATH", None)

    history = model.epoch_history.copy()
    model_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(model_dir / "history.csv", index=False)
    emit("train_complete", variant=variant, epochs=int(history["epoch"].nunique()))
    mark_done(variant, "training_complete")
    return model


# ==============================================================================
# Prediction (counterfactual: control cells -> predicted under drug)
# ==============================================================================

def predict_ood(model, adata, variant, args):
    """Predict OOD counterfactual expression for PW034_Panobinostat."""
    from cpa._utils import CPA_REGISTRY_KEYS

    # Get basal (control) cells of the target patient
    target_patient = "PW034"
    target_drug = "Panobinostat"
    target_dose = 1.0

    basal_mask = (
        adata.obs["cell_type"].eq(target_patient).to_numpy()
        & control_mask(adata.obs)
    )
    basal = adata[basal_mask].copy()
    emit("basal_extracted", variant=variant, n=int(basal.n_obs))

    set_counterfactual_obs(
        basal.obs,
        cell_type=target_patient,
        condition=target_drug,
        dose=float(target_dose),
    )
    basal.obs["split"] = "ood_predict"
    basal.obs["CPA_control"] = 0

    max_comb_len = int(CPA_REGISTRY_KEYS.MAX_COMB_LENGTH)
    perts = [model.pert_encoder[target_drug]] + [CPA_REGISTRY_KEYS.PADDING_IDX] * (max_comb_len - 1)
    doses = [float(target_dose)] + [0.0] * (max_comb_len - 1)
    basal.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS] = np.tile(np.asarray(perts, dtype=np.int64), (basal.n_obs, 1))
    basal.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS_DOSAGES] = np.tile(np.asarray(doses, dtype=np.float32), (basal.n_obs, 1))

    category_key = CPA_REGISTRY_KEYS.CATEGORY_KEY
    if category_key in basal.obs:
        basal.obs[category_key] = basal.obs[["cell_type", "condition"]].apply(
            lambda row: "_".join(row.astype(str)), axis=1)

    model.predict(basal, batch_size=args.batch_size, n_samples=1, return_mean=True)
    pred_x = np.asarray(basal.obsm["CPA_pred"], dtype=np.float32)
    pred_x = np.nan_to_num(pred_x, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    pred_x[pred_x < 0] = 0.0

    pred_path = DELIVERY / "predictions" / f"CPA_GBM_{variant}_pred.h5ad"
    pred = ad.AnnData(X=pred_x, obs=basal.obs.copy(), var=adata.var.copy(),
                      uns={"prediction": {"method": f"CPA_GBM_{variant}", "target_patient": target_patient,
                                          "target_drug": target_drug, "target_dose": target_dose}})
    pred.obs_names = basal.obs_names.copy()
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(pred_path, compression="gzip")
    emit("prediction_saved", variant=variant, output=str(pred_path), shape=list(pred.shape))
    return pred


# ==============================================================================
# Evaluation via evaluate_nips_gbm.py
# ==============================================================================

def run_evaluation(adata_path: Path, pred_path: Path, method: str, output_dir: Path):
    """Run NIPS evaluation protocol on GBM predictions."""
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
        "--setting", "ood",
        "--output-dir", str(output_dir),
    ]
    print(f"\nRunning NIPS evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"NIPS evaluation returned non-zero exit code: {result.returncode}")
    else:
        print("NIPS evaluation completed successfully.")


# ==============================================================================
# Main
# ==============================================================================

def run_variant(variant, CPA, adata, args):
    """Run a single CPA variant: train -> predict -> evaluate."""
    print(f"\n{'='*70}")
    print(f"  CPA Variant: {variant} (GBM)")
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
        aligned_cache = DELIVERY / "models" / f"CPA_GBM_{variant}_aligned.h5ad"
        aligner_file = _aligner_path(variant)

        if not args.fresh and aligned_cache.exists():
            print(f"\nLoading cached aligned AnnData: {aligned_cache}")
            run_adata = ad.read_h5ad(str(aligned_cache))
            print(f"  Aligned shape: {run_adata.shape}")
        elif not args.fresh and is_done(variant, "aligner_trained") and aligner_file.exists():
            print(f"\nLoading saved aligner: {aligner_file}")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_genes = adata.n_vars
            aligner = Aligner(input_dim=adata.obsm["X_scGPT"].shape[1], output_dim=n_genes).to(device)
            aligner.load_state_dict(torch.load(aligner_file, map_location=device))
            print("Building aligned AnnData...")
            run_adata = build_aligned_adata(adata, aligner, "X_scGPT", str(device))
            print(f"  Aligned shape: {run_adata.shape}")
            run_adata.write_h5ad(str(aligned_cache), compression="gzip")
            print(f"  Cached aligned AnnData: {aligned_cache}")
            mark_done(variant, "aligned_built")
        else:
            print("\nTraining scGPT aligner (512d -> 5000d gene space)...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            aligner = train_aligner(adata, scgpt_key="X_scGPT", device=str(device), epochs=30)
            torch.save(aligner.state_dict(), aligner_file)
            print(f"  Aligner saved: {aligner_file}")
            mark_done(variant, "aligner_trained")
            print("Building aligned AnnData...")
            run_adata = build_aligned_adata(adata, aligner, "X_scGPT", str(device))
            print(f"  Aligned shape: {run_adata.shape}")
            run_adata.write_h5ad(str(aligned_cache), compression="gzip")
            print(f"  Cached aligned AnnData: {aligned_cache}")
            mark_done(variant, "aligned_built")

    # Train
    pred_path = DELIVERY / "predictions" / f"CPA_GBM_{variant}_pred.h5ad"
    pred = None

    skip_train = args.skip_train or (not args.fresh and is_done(variant, "training_complete"))
    if skip_train:
        print(f"\nSkipping training (already completed).")
        model = None
    else:
        model = train_cpa_variant(CPA, run_adata, variant, args)

    # Predict
    if pred is None:
        if not args.fresh and pred_path.exists():
            print(f"\nLoading existing prediction: {pred_path}")
            pred = ad.read_h5ad(str(pred_path))
        else:
            if model is None:
                print("Loading existing model for prediction...")
                from cpa._model import CPA as CPAModel
                model_dir = DELIVERY / "models" / f"CPA_GBM_{variant}_model"
                model = CPAModel.load(model_dir, run_adata)
            pred = predict_ood(model, run_adata, variant, args)
    mark_done(variant, "prediction_saved")

    # Evaluate
    out_dir = DELIVERY / "evaluation" / "gbm_dataset"
    out_json = out_dir / f"CPA_GBM_{variant}_ood_metrics.json"
    skip_eval = args.skip_eval or (not args.fresh and is_done(variant, "eval_done") and out_json.exists())

    if skip_eval:
        print(f"\nSkipping evaluation (already completed).")
        if out_json.exists():
            print(f"  Results: {out_json}")
    else:
        run_evaluation(args.adata, pred_path, f"CPA_GBM_{variant}", out_dir)
        mark_done(variant, "eval_done")

    return True


def main():
    args = parse_args()
    log_path = DELIVERY / "logs" / "CPA_GBM_dataset_training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as lh, \
         contextlib.redirect_stdout(Tee(sys.stdout, lh)), \
         contextlib.redirect_stderr(Tee(sys.stderr, lh)):

        emit("run_start", variant=args.variant, fresh=args.fresh)

        variants = ["M0", "M1", "M4", "M5"] if args.variant == "all" else [args.variant]

        if args.fresh:
            clear_progress(variants)

        CPA = install_cpa_compat()

        print(f"Loading GBM dataset: {args.adata}")
        adata = ad.read_h5ad(args.adata)
        adata.obs = adata.obs.copy()
        ensure_nips_aliases(adata, add_legacy_aliases=True)

        if "control" not in adata.obs.columns:
            adata.obs["control"] = adata.obs["neg_control"].astype(int).eq(1)

        print(f"  Shape: {adata.shape}, splits: {adata.obs['split'].value_counts().to_dict()}")
        print(f"  DEGs: {len(adata.uns.get('rank_genes_groups_cov', {}))} groups")

        for v in variants:
            run_variant(v, CPA, adata, args)

        emit("run_complete")


if __name__ == "__main__":
    main()
