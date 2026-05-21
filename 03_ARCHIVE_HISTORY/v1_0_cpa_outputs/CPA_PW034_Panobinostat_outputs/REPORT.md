# CPA OOD Prediction Report: PW034 + Panobinostat

## 1. Objective

This run trained a Compositional Perturbation Autoencoder (CPA) on the prepared GBM perturbation AnnData object and evaluated zero-shot out-of-distribution prediction for:

- Target patient/covariate: `PW034`
- Source/basal state: `PW034` + `control`
- Target perturbation: `Panobinostat`
- Target dosage: `1.0`

The requested output prediction is stored as an AnnData object with `adata.X` containing the CPA predicted mean expression.

## 2. Input Data

Input file:

- `GBM_Universal_Perturbation_Ready.h5ad`

AnnData summary:

- Cells: `169,972`
- Genes/features: `5,000`
- Count layer: `adata.layers['counts']`
- Primary expression matrix used for downstream metric comparison: `adata.X`
- Split column: `adata.obs['split']`
- Perturbation column: `adata.obs['perturbation']`
- Dosage column: `adata.obs['dosage']`
- Patient/covariate column: `adata.obs['covariate_patient']`
- Compatible patient alias used by the evaluator: `adata.obs['cell_type']`

Split composition:

| Split | Cells |
|---|---:|
| train | 150,564 |
| valid | 16,729 |
| ood | 2,679 |

Target cells:

| Group | Filter | Cells |
|---|---|---:|
| Basal input | `covariate_patient == 'PW034'` and `perturbation == 'control'` | 15,288 |
| Ground truth OOD | `covariate_patient == 'PW034'` and `perturbation == 'Panobinostat'` | 2,679 |

The count layer was checked for positive library sizes before training. Count library totals ranged from `8.0` to `50,481.0`, with mean `979.857`.

## 3. CPA Setup

The full AnnData object was registered with CPA using the requested API mapping:

| CPA setup field | Value |
|---|---|
| `perturbation_key` | `perturbation` |
| `control_group` | `control` |
| `dosage_key` | `dosage` |
| `categorical_covariate_keys` | `['covariate_patient']` |
| `layer` | `counts` |
| `is_count_data` | `True` |

The strict train/validation boundaries were respected through:

- `split_key='split'`
- `train_split='train'`
- `valid_split='valid'`
- `test_split='ood'`

GPU use was restricted to the allowed shared-server range. The full run used:

- `CUDA_VISIBLE_DEVICES=4`

No other GPU processes were stopped or modified.

## 4. Model Training

Model:

- CPA (`cpa.CPA`)
- Latent dimension: `n_latent=32`
- Reconstruction loss: Negative Binomial (`recon_loss='nb'`)
- Batch size: `1024`
- Maximum epochs: `50`
- Early stopping patience: `20`
- Random seed: `7`

The run reached the configured maximum of 50 epochs. The training loss convergence was logged to:

- `GBM_CPA_training.log`
- `GBM_CPA_model/history.csv`
- `GBM_CPA_model/epoch_history.tsv`

Best validation summary by the CPA validation selection metric in the runner:

| Epoch | Validation recon loss | Validation R2 mean | Validation R2 var | CPA metric |
|---:|---:|---:|---:|---:|
| 45 | 909.863 | 0.8566 | 0.7103 | 2.6462 |

Final logged validation row:

| Epoch | Validation recon loss | Validation R2 mean | Validation R2 var |
|---:|---:|---:|---:|
| 49 | 910.650 | 0.8556 | 0.7134 |

CPA emitted repeated validation warnings about variance degrees of freedom and empty means inside its internal validation metric code. These warnings did not stop training, prediction, or evaluation.

## 5. OOD Prediction

Prediction input:

- All `PW034` control cells (`15,288` cells)

Counterfactual target assigned for CPA prediction:

- `perturbation = 'Panobinostat'`
- `dosage = 1.0`

Output file:

- `GBM_CPA_PW034_Panobinostat_pred.h5ad`

Prediction AnnData validation:

| Field | Value |
|---|---:|
| Shape | `(15,288, 5,000)` |
| Patient labels | all `PW034` |
| Perturbation labels | all `Panobinostat` |
| `X` finite | `True` |
| `X` min | `1.3438e-08` |
| `X` mean | `0.09739` |
| `X` max | `656.869` |

No post hoc expression re-normalization was applied. The only numerical safety handling was replacing non-finite values if present and clipping negative predictions to zero; the final matrix was finite and non-negative.

## 6. Evaluation

Evaluation target:

- Ground truth `PW034` + `Panobinostat` cells from `GBM_Universal_Perturbation_Ready.h5ad`

Gene set:

- Strictly the pre-existing 50 genes in `adata.uns['top50_DEGs']['PW034|Panobinostat']`

Top 10 genes from the evaluation set:

`VNN2`, `KRT17`, `HES5`, `GABRA3`, `SPON1`, `MXRA5`, `LINC01341`, `VNN1`, `PPP1R36`, `SECTM1`

Metrics were computed using the same definitions as the existing `scripts/evaluate_crisp_ood.py` logic:

- Pearson correlation of predicted vs. true perturbation delta over Top50 DE genes
- Spearman correlation of predicted vs. true perturbation delta over Top50 DE genes
- R2 score on predicted vs. true post-perturbation mean expression over Top50 DE genes
- Sinkhorn distance between predicted and true Top50 expression distributions
- Direction accuracy of predicted perturbation delta signs over Top50 DE genes

## 7. Results

The CPA row was appended/upserted into `GBM_CRISP_OOD_metrics.md`.

| Method | Target Covariate (Patient) | Target Drug | PrΔ DE (↑) | Sp DE (↑) | R² score DE (↑) | Sinkhorn DE (↓) | Direction Accuracy (%) (↑) |
|---|---|---|---:|---:|---:|---:|---:|
| MeanShiftBaseline | PW034 | Panobinostat | -0.468 | -0.230 | -276.429 | 0.190 | 20.0% |
| CPA | PW034 | Panobinostat | 0.608 | 0.463 | -18.636 | 0.004 | 96.0% |

Interpretation:

- CPA substantially improved perturbation-direction recovery relative to the mean-shift baseline: `96.0%` vs. `20.0%`.
- CPA improved Top50 delta correlation: Pearson `0.608` and Spearman `0.463`, compared with negative correlations for the baseline.
- CPA reduced Sinkhorn distance strongly: `0.004` vs. `0.190`.
- The R2 score remained negative (`-18.636`), indicating that although CPA captured the direction/ranking of many DE shifts, absolute post-treatment mean expression over the Top50 genes was still not calibrated enough to match the ground truth means under this metric.

## 8. Output Directory Contents

This directory organizes the run outputs. Large artifacts are linked instead of duplicated.

| Path | Description |
|---|---|
| `REPORT.md` | This report |
| `GBM_CPA_PW034_Panobinostat_pred.h5ad` | Symlink to CPA OOD prediction AnnData |
| `GBM_CRISP_OOD_metrics.md` | Symlink to updated metrics markdown |
| `GBM_CPA_training.log` | Symlink to full training and prediction log |
| `history.csv` | Symlink to CPA epoch history CSV |
| `epoch_history.tsv` | Symlink to CPA epoch history TSV |
| `GBM_CPA_model` | Symlink to saved CPA model directory |
| `train_cpa_ood.py` | Symlink to the runner used for training, prediction, and metric upsert |

