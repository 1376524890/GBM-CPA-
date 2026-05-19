#!/usr/bin/env python
"""Train CPA with scGPT-aligned cells + MolFormer drug embeddings (M5).

Combines:
  - Cell side: scGPT 512d → MLP Aligner → 5000d aligned expression
  - Drug side: MolFormer 768d → Linear → 32d latent (frozen)

Usage:
  conda activate plknature
  CUDA_VISIBLE_DEVICES=0 python scripts/train_cpa_scgpt_molformer.py
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
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
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--adata", type=Path, default=ROOT / "GBM_NIPS_Ready.h5ad")
    parser.add_argument("--aligner", type=Path, default=ROOT / "GBM_scGPT_aligner.pt")
    parser.add_argument("--molformer-parquet", type=Path, default=ROOT / "GBM_molformer_drug_emb.parquet")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "GBM_CPA_scGPT_MolFormer_model")
    parser.add_argument("--predicted", type=Path, default=ROOT / "GBM_CPA_scGPT_MolFormer_PW034_Panobinostat_pred.h5ad")
    parser.add_argument("--target-patient", default="PW034")
    parser.add_argument("--target-drug", default="Panobinostat")
    parser.add_argument("--target-dosage", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class Aligner(nn.Module):
    def __init__(self, input_dim=512, output_dim=5000, hidden_dims=(1024, 2048, 4096), dropout=0.1):
        super().__init__()
        layers = []; prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def install_cpa_compat():
    from scvi.model._utils import parse_device_args
    from scvi import settings as scvi_settings
    import scvi.model._utils as smu, scvi.train as st, scvi.train._callbacks as stc
    from lightning.pytorch.callbacks import Callback
    from scvi.train import TrainRunner as CTR
    def puga(use_gpu=None, return_device=False):
        if use_gpu is None or use_gpu == "auto": acc, dev = "auto", "auto"
        elif use_gpu is False: acc, dev = "cpu", "auto"
        elif use_gpu is True: acc, dev = "cuda", "auto"
        elif isinstance(use_gpu, int): acc, dev = "cuda", [use_gpu]
        elif isinstance(use_gpu, str):
            v = use_gpu.lower()
            if v in {"cuda","gpu"}: acc, dev = "cuda", "auto"
            elif v == "cpu": acc, dev = "cpu", "auto"
            else: acc, dev = "cuda", [int(use_gpu)]
        else: acc, dev = "auto", "auto"
        return parse_device_args(accelerator=acc, devices=dev, return_device="torch") if return_device else (acc, dev)
    class SBS(Callback):
        def __init__(self, monitor="validation_loss", mode="min", period=1, verbose=False, **kw):
            super().__init__(); self.monitor=monitor; self.mode=mode; self.period=period; self.verbose=verbose
    if not hasattr(scvi_settings,"dl_pin_memory_gpu_training"): scvi_settings.dl_pin_memory_gpu_training=False
    smu.parse_use_gpu_arg=puga; stc.SaveBestState=SBS
    class CTR2(CTR):
        def __init__(self, model, training_plan, data_splitter, max_epochs, accelerator=None, devices=None, use_gpu=None, **kw):
            if accelerator is None: accelerator, pd2 = puga(use_gpu=use_gpu, return_device=False); devices=pd2 if devices is None else devices
            if devices is None: devices="auto"
            super().__init__(model=model, training_plan=training_plan, data_splitter=data_splitter, max_epochs=max_epochs, accelerator=accelerator, devices=devices, **kw)
    st.TrainRunner=CTR2
    def patch(tc):
        if getattr(tc,"_c2c",False): return
        ots, ovs, otee, ovee = tc.training_step, tc.validation_step, tc.training_epoch_end, tc.validation_epoch_end
        def ts(self,*a,**kw): o=ots(self,*a,**kw); self._cto.append(o); return o
        def vs(self,*a,**kw): o=ovs(self,*a,**kw); self._cvo.append(o); return o
        def otes(self): self._cto=[]
        def oves(self): self._cvo=[]
        def otee2(self):
            o=getattr(self,"_cto",[])
            if o: otee(self,o)
            self._cto=[]
        def ovee2(self):
            o=getattr(self,"_cvo",[])
            if o: ovee(self,o)
            self._cvo=[]
        tc.training_step=ts; tc.validation_step=vs; tc.on_train_epoch_start=otes; tc.on_validation_epoch_start=oves
        tc.on_train_epoch_end=otee2; tc.on_validation_epoch_end=ovee2
        delattr(tc,"training_epoch_end"); delattr(tc,"validation_epoch_end"); tc._c2c=True
    spec=importlib.util.find_spec("cpa"); pd2=Path(next(iter(spec.submodule_search_locations)))
    pkg=types.ModuleType("cpa"); pkg.__path__=[str(pd2)]; sys.modules["cpa"]=pkg
    ms=importlib.util.spec_from_file_location("cpa._model", pd2/"_model.py")
    mm=importlib.util.module_from_spec(ms); sys.modules["cpa._model"]=mm; ms.loader.exec_module(mm)
    from cpa._task import CPATrainingPlan; patch(CPATrainingPlan)
    return mm.CPA


def reset_cpa(CPA): CPA.pert_encoder=None; CPA.covars_encoder=None; CPA.pert_smiles_map=None


def build_molformer_emb(parquet_path, pert_encoder):
    drug_df = pd.read_parquet(parquet_path)
    mdim = drug_df.shape[1]
    s2e = {smi: drug_df.loc[smi].values.astype(np.float32) for smi in drug_df.index}
    n_perts = len(pert_encoder)
    emat = np.zeros((n_perts, mdim), dtype=np.float32)
    adata = ad.read_h5ad(ROOT / "GBM_Universal_Perturbation_Ready.h5ad", backed="r")
    ds = adata.uns.get("drug_smiles", {})
    for dname, idx in pert_encoder.items():
        if dname in ("<PAD>", "control"): continue
        smi = ds.get(dname, "")
        if smi in s2e: emat[idx] = s2e[smi]
    from cpa._utils import CPA_REGISTRY_KEYS
    emb = torch.nn.Embedding(n_perts, mdim, padding_idx=CPA_REGISTRY_KEYS.PADDING_IDX)
    emb.weight.data.copy_(torch.tensor(emat)); emb.weight.requires_grad = False
    return emb


def main():
    args = parse_args()
    log_path = args.predicted.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as lh, contextlib.redirect_stdout(Tee(sys.stdout, lh)), contextlib.redirect_stderr(Tee(sys.stderr, lh)):
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"M5: scGPT + MolFormer, Device: {device}")

        # 1. Build aligned AnnData (scGPT → gene space)
        print("Building aligned AnnData...")
        adata_raw = ad.read_h5ad(args.adata)
        for col in ["perturbation","covariate_patient","cell_type"]:
            adata_raw.obs[col] = adata_raw.obs[col].astype(str)
        adata_raw.obs["dosage"] = pd.to_numeric(adata_raw.obs["dosage"], errors="raise").astype(float).astype(str)

        ckpt = torch.load(args.aligner, map_location="cpu")
        aligner = Aligner(input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"])
        aligner.load_state_dict(ckpt["model_state_dict"]); aligner.to(device); aligner.eval()

        X_scgpt = adata_raw.obsm["X_scGPT"].astype(np.float32)
        aligned = np.zeros((adata_raw.n_obs, ckpt["output_dim"]), dtype=np.float32)
        bs = 512
        with torch.no_grad():
            for start in range(0, adata_raw.n_obs, bs):
                end = min(start+bs, adata_raw.n_obs)
                aligned[start:end] = aligner(torch.tensor(X_scgpt[start:end], dtype=torch.float32).to(device)).cpu().numpy()
        aligned = np.clip(aligned, 0, None)

        adata = ad.AnnData(X=aligned, obs=adata_raw.obs.copy(), var=adata_raw.var.copy(), uns=adata_raw.uns.copy())
        adata.layers["counts"] = aligned.copy()
        adata.obsm = {k: v.copy() for k, v in adata_raw.obsm.items()}
        del adata_raw, aligned, X_scgpt
        print(f"  Aligned shape: {adata.shape}")

        # 2. Setup CPA with MolFormer drug embeddings
        reset_cpa(CPA := install_cpa_compat())
        CPA.setup_anndata(adata, perturbation_key="perturbation", control_group="control",
                          dosage_key="dosage", categorical_covariate_keys=["covariate_patient"],
                          layer="counts", is_count_data=False)
        molformer_emb = build_molformer_emb(args.molformer_parquet, CPA.pert_encoder)

        # 3. Train
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        use_gpu = "auto"
        model = CPA(adata, split_key="split", train_split="train", valid_split="valid", test_split="ood",
                    n_latent=32, recon_loss="gauss", seed=args.seed, drug_embeddings=molformer_emb)

        if args.model_dir.exists():
            bk = args.model_dir.with_name(f"{args.model_dir.name}.previous")
            if bk.exists(): shutil.rmtree(bk)
            args.model_dir.rename(bk)

        print("Training M5...")
        model.train(max_epochs=args.max_epochs, use_gpu=use_gpu, batch_size=args.batch_size,
                    save_path=str(args.model_dir), check_val_every_n_epoch=1,
                    early_stopping_patience=args.early_stopping_patience,
                    plan_kwargs={"do_clip_grad":True, "gradient_clip_value":3.0, "n_epochs_verbose":1},
                    log_every_n_steps=25, enable_progress_bar=True)

        history = model.epoch_history.copy()
        args.model_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.model_dir/"history.csv", index=False)
        valid = history[history["mode"].eq("valid")].copy()
        if not valid.empty:
            valid["cpa_metric"] = valid["r2_mean"]+0.5*valid["r2_var"]+np.exp(valid["disnt_after"]-valid["disnt_basal"])
            best = valid.loc[valid["cpa_metric"].idxmax()]
            print(f"Best epoch {int(best['epoch'])}: r2_mean={best['r2_mean']:.4f}, r2_var={best['r2_var']:.4f}")

        # 4. Predict
        from cpa._utils import CPA_REGISTRY_KEYS
        basal_mask = (adata.obs["covariate_patient"].eq(args.target_patient) & adata.obs["perturbation"].eq("control")).to_numpy()
        basal = adata[basal_mask].copy()
        basal.obs["perturbation"] = args.target_drug
        basal.obs["dosage"] = str(float(args.target_dosage))
        basal.obs["is_control"] = False
        basal.obs["split"] = "ood_predict"
        basal.obs["CPA_control"] = 0

        mcl = int(CPA_REGISTRY_KEYS.MAX_COMB_LENGTH)
        basal.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS] = np.tile(np.asarray([model.pert_encoder[args.target_drug]]+[CPA_REGISTRY_KEYS.PADDING_IDX]*(mcl-1), dtype=np.int64), (basal.n_obs, 1))
        basal.obsm[CPA_REGISTRY_KEYS.PERTURBATIONS_DOSAGES] = np.tile(np.asarray([float(args.target_dosage)]+[0.0]*(mcl-1), dtype=np.float32), (basal.n_obs, 1))
        if (ck2 := CPA_REGISTRY_KEYS.CATEGORY_KEY) in basal.obs:
            basal.obs[ck2] = basal.obs[["covariate_patient","perturbation"]].apply(lambda r: "_".join(r.astype(str)), axis=1)

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
