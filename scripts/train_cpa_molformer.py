#!/usr/bin/env python
"""Train CPA with MolFormer drug embeddings (M4) and evaluate OOD prediction.

This is a variant of train_cpa_ood.py that uses pre-computed MolFormer (768d)
drug embeddings instead of CPA's learnable pert_encoder embeddings.

Usage:
  conda activate plknature
  CUDA_VISIBLE_DEVICES=0 python scripts/train_cpa_molformer.py
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
from scipy import sparse, stats
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]

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
    parser.add_argument("--adata", type=Path, default=ROOT / "GBM_Universal_Perturbation_Ready.h5ad")
    parser.add_argument("--molformer-parquet", type=Path, default=ROOT / "GBM_molformer_drug_emb.parquet")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "GBM_CPA_MolFormer_model")
    parser.add_argument("--predicted", type=Path, default=ROOT / "GBM_CPA_MolFormer_PW034_Panobinostat_pred.h5ad")
    parser.add_argument("--metrics-md", type=Path, default=ROOT / "GBM_CRISP_OOD_metrics.md")
    parser.add_argument("--training-log", type=Path, default=ROOT / "GBM_CPA_MolFormer_training.log")
    parser.add_argument("--target-patient", default="PW034")
    parser.add_argument("--target-drug", default="Panobinostat")
    parser.add_argument("--target-dosage", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sinkhorn-samples", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
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
        parsed = parse_device_args(accelerator=accelerator, devices=devices, return_device="torch")
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
        def __init__(self, model, training_plan, data_splitter, max_epochs,
                     accelerator=None, devices=None, use_gpu=None, **trainer_kwargs):
            if accelerator is None:
                accelerator, parsed_devices = parse_use_gpu_arg(use_gpu=use_gpu, return_device=False)
                devices = parsed_devices if devices is None else devices
            if devices is None:
                devices = "auto"
            super().__init__(model=model, training_plan=training_plan, data_splitter=data_splitter,
                             max_epochs=max_epochs, accelerator=accelerator, devices=devices, **trainer_kwargs)

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


def build_molformer_drug_embeddings(parquet_path: Path, pert_encoder: dict):
    """Build nn.Embedding with MolFormer weights aligned to pert_encoder ordering.

    Returns (embedding, molformer_dim).
    """
    drug_emb_df = pd.read_parquet(parquet_path)
    molformer_dim = drug_emb_df.shape[1]  # 768

    smiles_to_emb = {}
    for smi in drug_emb_df.index:
        smiles_to_emb[smi] = drug_emb_df.loc[smi].values.astype(np.float32)

    n_perts = len(pert_encoder)
    embed_matrix = np.zeros((n_perts, molformer_dim), dtype=np.float32)

    # Get drug->SMILES mapping from the AnnData
    from rdkit import Chem
    adata = ad.read_h5ad(ROOT / "GBM_Universal_Perturbation_Ready.h5ad", backed="r")
    drug_smiles = adata.uns.get("drug_smiles", {})

    for drug_name, idx in pert_encoder.items():
        if drug_name in ("<PAD>", "control"):
            continue
        smi = drug_smiles.get(drug_name, "")
        # Try exact match first, then canonical
        if smi in smiles_to_emb:
            embed_matrix[idx] = smiles_to_emb[smi]
        elif smi:
            canonical = Chem.CanonSmiles(smi)
            if canonical in smiles_to_emb:
                embed_matrix[idx] = smiles_to_emb[canonical]

    from cpa._utils import CPA_REGISTRY_KEYS
    embedding = torch.nn.Embedding(n_perts, molformer_dim, padding_idx=CPA_REGISTRY_KEYS.PADDING_IDX)
    embedding.weight.data.copy_(torch.tensor(embed_matrix))
    embedding.weight.requires_grad = False  # Freeze MolFormer embeddings

    emit("molformer_embeddings_built", n_perts=n_perts, molformer_dim=molformer_dim,
         non_zero_rows=int((embed_matrix.sum(axis=1) != 0).sum()))
    return embedding, molformer_dim


def validate_adata(adata: ad.AnnData, patient: str, drug: str) -> None:
    required_obs = {"perturbation", "dosage", "covariate_patient", "split", "is_control"}
    missing = sorted(required_obs.difference(adata.obs.columns))
    if missing:
        raise ValueError(f"Missing required obs columns: {missing}")
    if "counts" not in adata.layers:
        raise ValueError("Expected integer counts in adata.layers['counts']")


def reset_cpa_class_state(CPA: type) -> None:
    CPA.pert_encoder = None
    CPA.covars_encoder = None
    CPA.pert_smiles_map = None


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


def compute_metrics(adata: ad.AnnData, pred: ad.AnnData, args: argparse.Namespace,
                    method_label: str) -> dict:
    key = f"{args.target_patient}|{args.target_drug}"
    genes = list(adata.uns["top50_DEGs"][key])
    gene_idx = adata.var_names.get_indexer(genes)
    pred_idx = pred.var_names.get_indexer(genes)

    true_mask = (adata.obs["cell_type"].eq(args.target_patient) & adata.obs["perturbation"].eq(args.target_drug)).to_numpy()
    ctrl_mask = (adata.obs["cell_type"].eq(args.target_patient) & adata.obs["is_control"]).to_numpy()

    x_true = dense(adata.X[true_mask][:, gene_idx])
    x_ctrl = dense(adata.X[ctrl_mask][:, gene_idx])
    x_pred = dense(pred.X[:, pred_idx])

    true_post = x_true.mean(axis=0)
    pred_post = x_pred.mean(axis=0)
    ctrl = x_ctrl.mean(axis=0)
    true_logfc = true_post - ctrl
    pred_logfc = pred_post - ctrl

    metrics = {
        "method": method_label,
        "patient": args.target_patient,
        "drug": args.target_drug,
        "pearson": float(stats.pearsonr(pred_logfc, true_logfc).statistic),
        "spearman": float(stats.spearmanr(pred_logfc, true_logfc).statistic),
        "r2": float(r2_score(true_post, pred_post)),
        "sinkhorn": sinkhorn_distance(x_pred, x_true, args.sinkhorn_samples, args.seed),
        "direction": float(np.mean(np.sign(pred_logfc) == np.sign(true_logfc)) * 100.0),
    }
    return metrics


def upsert_metrics_row(metrics_path: Path, metrics: dict) -> None:
    header = (
        "| Method | Target Covariate (Patient) | Target Drug | PrΔ DE (↑) | Sp DE (↑) | R² score DE (↑) | Sinkhorn DE (↓) | Direction Accuracy (%) (↑) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    row = (f"| {metrics['method']} | {metrics['patient']} | {metrics['drug']} | "
           f"{metrics['pearson']:.3f} | {metrics['spearman']:.3f} | {metrics['r2']:.3f} | "
           f"{metrics['sinkhorn']:.3f} | {metrics['direction']:.1f}% |")

    if metrics_path.exists():
        lines = metrics_path.read_text().splitlines()
        if len(lines) < 2 or not lines[0].startswith("| Method |"):
            lines = header.rstrip("\n").splitlines()
    else:
        lines = header.rstrip("\n").splitlines()

    filtered = []
    for line in lines:
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        is_same_row = (len(parts) >= 3 and parts[0] == metrics["method"]
                       and parts[1] == metrics["patient"] and parts[2] == metrics["drug"])
        if not is_same_row:
            filtered.append(line)
    filtered.append(row)
    metrics_path.write_text("\n".join(filtered).rstrip() + "\n")
    emit("metrics_md_updated", output=str(metrics_path), row=row)


def main() -> None:
    args = parse_args()
    args.training_log.parent.mkdir(parents=True, exist_ok=True)

    with args.training_log.open("w") as log_handle, \
         contextlib.redirect_stdout(Tee(sys.stdout, log_handle)), \
         contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):

        emit("run_start", method="CPA+MolFormer (M4)",
             args={k: str(v) for k, v in vars(args).items()})
        os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

        # Install CPA compatibility layer
        CPA = install_cpa_compat()
        emit("cpa_loaded")

        # Load and prepare AnnData
        adata = ad.read_h5ad(args.adata)
        adata.obs = adata.obs.copy()
        adata.obs["perturbation"] = adata.obs["perturbation"].astype(str)
        adata.obs["covariate_patient"] = adata.obs["covariate_patient"].astype(str)
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
        adata.obs["dosage"] = pd.to_numeric(adata.obs["dosage"], errors="raise").astype(float).astype(str)
        validate_adata(adata, args.target_patient, args.target_drug)

        # Setup CPA to get pert_encoder, then build MolFormer embeddings
        reset_cpa_class_state(CPA)
        CPA.setup_anndata(
            adata,
            perturbation_key="perturbation",
            control_group="control",
            dosage_key="dosage",
            categorical_covariate_keys=["covariate_patient"],
            layer="counts",
            is_count_data=True,
        )
        emit("setup_complete", pert_encoder=CPA.pert_encoder)

        # Build MolFormer drug embeddings
        molformer_emb, molformer_dim = build_molformer_drug_embeddings(
            args.molformer_parquet, CPA.pert_encoder)

        # Reset and re-setup for fresh training
        reset_cpa_class_state(CPA)
        CPA.setup_anndata(
            adata,
            perturbation_key="perturbation",
            control_group="control",
            dosage_key="dosage",
            categorical_covariate_keys=["covariate_patient"],
            layer="counts",
            is_count_data=True,
        )

        # Train CPA with MolFormer embeddings
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
            drug_embeddings=molformer_emb,
        )
        emit("model_created", drug_embedding_dim=molformer_dim)

        if args.model_dir.exists():
            backup = args.model_dir.with_name(f"{args.model_dir.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            args.model_dir.rename(backup)

        emit("train_start", max_epochs=args.max_epochs, batch_size=args.batch_size)
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

        history = model.epoch_history.copy()
        args.model_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.model_dir / "history.csv", index=False)
        history.to_csv(args.model_dir / "epoch_history.tsv", sep="\t", index=False)

        valid = history[history["mode"].eq("valid")].copy()
        best = {}
        if not valid.empty:
            valid["cpa_metric"] = valid["r2_mean"] + 0.5 * valid["r2_var"] + np.exp(valid["disnt_after"] - valid["disnt_basal"])
            best_row = valid.loc[valid["cpa_metric"].idxmax()]
            best = {"epoch": int(best_row["epoch"]), "val_recon": float(best_row["recon_loss"]),
                    "val_r2_mean": float(best_row["r2_mean"]), "val_r2_var": float(best_row["r2_var"]),
                    "cpa_metric": float(best_row["cpa_metric"])}
        emit("train_complete", best_valid=best)

        # Generate OOD predictions
        from cpa._utils import CPA_REGISTRY_KEYS
        basal_mask = (adata.obs["covariate_patient"].eq(args.target_patient)
                      & adata.obs["perturbation"].eq("control")).to_numpy()
        basal = adata[basal_mask].copy()
        emit("basal_extracted", n=int(basal.n_obs))

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
            basal.obs[category_key] = basal.obs[["covariate_patient", "perturbation"]].apply(
                lambda row: "_".join(row.astype(str)), axis=1)

        model.predict(basal, batch_size=args.batch_size, n_samples=1, return_mean=True)
        pred_x = np.asarray(basal.obsm["CPA_pred"], dtype=np.float32)
        pred_x = np.nan_to_num(pred_x, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
        pred_x[pred_x < 0] = 0.0

        pred = ad.AnnData(X=pred_x, obs=basal.obs.copy(), var=adata.var.copy(),
                           uns={"prediction": {"method": "CPA+MolFormer (M4)", "target_patient": args.target_patient,
                                               "target_drug": args.target_drug, "target_dosage": float(args.target_dosage),
                                               "x_semantics": "CPA predicted mean expression using MolFormer drug embeddings."}})
        pred.obs_names = basal.obs_names.copy()
        args.predicted.parent.mkdir(parents=True, exist_ok=True)
        pred.write_h5ad(args.predicted, compression="gzip")
        emit("prediction_saved", output=str(args.predicted), shape=list(pred.shape))

        # Compute and save metrics
        metrics = compute_metrics(adata, pred, args, "CPA (M4: +MolFormer)")
        upsert_metrics_row(args.metrics_md, metrics)
        emit("run_complete", metrics=metrics)


if __name__ == "__main__":
    main()
