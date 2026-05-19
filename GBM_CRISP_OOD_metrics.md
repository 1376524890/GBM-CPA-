| Method | Target Covariate (Patient) | Target Drug | PrΔ DE (↑) | Sp DE (↑) | R² score DE (↑) | Sinkhorn DE (↓) | Direction Accuracy (%) (↑) |
|---|---|---|---|---|---|---|---|
| MeanShiftBaseline | PW034 | Panobinostat | -0.468 | -0.230 | -276.429 | 0.190 | 20.0% |
| MLP (M1: scGPT) | PW034 | Panobinostat | 0.018 | -0.003 | -1046.680 | 0.012 | 18.0% |
| MLP (M5: scGPT+MolFormer) | PW034 | Panobinostat | -0.069 | -0.086 | -1017.322 | 0.013 | 20.0% |
| CPA (M0: baseline) | PW034 | Panobinostat | 0.608 | 0.463 | -18.636 | 0.004 | 96.0% |
| CPA (M4: +MolFormer) | PW034 | Panobinostat | 0.693 | 0.585 | -18.957 | 0.004 | 94.0% |
| CPA (M1: +scGPT all) | PW034 | Panobinostat | 0.103 | 0.219 | -311.580 | 0.006 | 36.0% |
| CPA (M2: +scGPT ctrl) | PW034 | Panobinostat | 0.111 | 0.173 | -486.665 | 0.006 | 44.0% |
| CPA (M3: +scGPT pert) | PW034 | Panobinostat | 0.027 | 0.209 | -929.834 | 0.009 | 46.0% |
| CPA (M5: scGPT+MolFormer) | PW034 | Panobinostat | 0.098 | 0.199 | -507.680 | 0.007 | 44.0% |
