# Domain adaptation to synthetic dark-skin dermoscopic images: lesion-level and zero-shot external evaluation

Code, per-image predictions and fold-level results for the article

> N. Sánchez-Medel, R. Díaz-Hernández, E. Ruiz-Muñoz, S. Zapotecas-Martínez, L. Altamirano-Robles.
> *Evaluating Domain Adaptation on Synthetic Dark-Skin Dermoscopic Images: Internal Gains Do Not Transfer to Real Clinical Cohorts* (under review).

The study adapts MobileNetV2 and EfficientNetB0 from HAM10000 to the style-transfer-derived Brown Skin dataset, evaluates every configuration under a lesion-level cross-validation that keeps a lesion and all its synthetic derivatives in the same fold, and then evaluates frozen final models zero-shot on four real cohorts (DDI, DDI-2, Fitzpatrick17k, PAD-UFES-20).

## Repository layout

```
pipeline/    training, final models and zero-shot evaluation
analysis/    statistics reported in the article
figures/     scripts that draw the figures
results/     fold-level results and per-image predictions used in the article
```

## Requirements

Python 3.10+ and the packages in `requirements.txt` (TensorFlow/Keras, scikit-learn, pandas, numpy, scipy, matplotlib). All models were trained on CPU.

## Data

Images are **not** redistributed. Download them from their official sources:

| Dataset | Role | Source |
|---|---|---|
| HAM10000 (+ `HAM10000_metadata.csv`) | source domain | Tschandl et al., *Sci. Data* 2018 (Harvard Dataverse / ISIC) |
| Brown Skin | synthetic target | Sánchez-Medel et al., *Computación y Sistemas* 2026; available from the corresponding author on reasonable request, or regenerated from HAM10000 with the published procedure |
| DDI (+ `ddi_metadata.csv`) | external | Daneshjou et al., *Sci. Adv.* 2022 (Stanford AIMI) |
| DDI-2 | external | Chang et al., *J. Invest. Dermatol.* 2024 |
| Fitzpatrick17k (FST IV–VI, benign/malignant) | external | Groh et al., CVPRW 2021 |
| PAD-UFES-20 | external | Pacheco et al., *Data in Brief* 2020 |

The pipeline expects this layout next to the scripts (each class folder holds the images of that class):

```
HAM10000_metadata.csv
datos_piel/
  pieles_claras/{BENIGNAS,MALIGNAS}/          HAM10000 (9,672 decodable images)
  pieles_oscuras/{BENIGNAS,MALIGNAS}/         Brown Skin (file names keep the HAM10000 image_id)
  DDI_clasificado/{BENIGNAS,MALIGNAS}/
  DDI2_clasificado/{BENIGNAS,MALIGNAS}/
  Fitzpatrick_clasificado/{BENIGNAS,MALIGNAS}/
  PADUFES_clasificado/{BENIGNAS,MALIGNAS}/    malignant: BCC, SCC, MEL; benign: ACK, NEV, SEK
```

## 1. Lesion-level cross-validation (`pipeline/`)

`entrenar_leakage_safe_v4.py` maps every Brown Skin image to its HAM10000 `lesion_id`, pools both datasets and partitions them with `StratifiedGroupKFold` (5 folds × 3 repeats; partition seed 2026 + r, training seed 2026 + 100r + f). The fold assignment of every image is written to `resultados/leakage_safe/split_ledger.csv`.

```
python entrenar_leakage_safe_v4.py --model A   --arch mobilenetv2      # A, B, C10, C20, C40, C60, C80, D, E
python entrenar_leakage_safe_v4.py --model C40 --arch efficientnetb0   # A, B, C40, D, E
```

Each fold appends one row to `resultados/leakage_safe/resultados_leakage_safe.csv` and saves the raw validation scores (`predicciones/pred_<model>[_efficientnetb0]_rep<r>_fold<f>.csv`). Runs already in the CSV are skipped.

## 2. Final models and zero-shot evaluation (`pipeline/`)

```
python entrenar_leakage_safe_v4.py --model C40 --train_final --arch mobilenetv2 [--final_seed 2027]
python correr_zeroshot_lote.py
```

`--train_final` trains one model on a lesion-level 85 % development split (the same split for every model and seed), freezes the Youden threshold on Brown Skin validation images, and evaluates the Brown Skin images of the 15 % hold-out as a same-distribution control. `correr_zeroshot_lote.py` evaluates every `FINAL_*.json` on the four cohorts (AUC, sensitivity and specificity at the frozen and at a re-optimized threshold, bootstrap CIs). `correr_fase2.bat` runs the full set used in the article on Windows: A and B (seed 2026) and C40 (seeds 2026, 2027, 2028) for both backbones.

## 3. Statistics (`analysis/`)

| Script | Input | Output in the article |
|---|---|---|
| `comparaciones_pareadas.py` | `results/fold_level_results.csv` | MobileNetV2 lesion-level summary and paired comparisons (corrected repeated-CV *t*-test, Holm) |
| `diferencias_por_repeticion.py` | `results/fold_level_results.csv` | per-repeat sensitivity analysis of those comparisons |
| `analisis_arquitecturas.py` | `results/fold_level_results.csv` | both-backbone summary; EfficientNetB0 and between-backbone paired comparisons |
| `analisis_zeroshot_dos_modelos.py` | `results/zeroshot/FINAL_C40_mobilenetv2`, `.../FINAL_C40_efficientnetb0` | zero-shot CIs (patients resampled for PAD-UFES-20) and EfficientNetB0 − MobileNetV2 differences |
| `analisis_fase2.py` | `results/zeroshot/` | A, B and C40 (3 seeds) external AUCs and paired A − C40 differences (Holm over 8 comparisons) |
| `analisis_ddi_fototipo.py` | `ddi_metadata.csv` + DDI predictions | DDI AUC by Fitzpatrick group, all positives and MEL/BCC/SCC only |

Examples:

```
python analysis/comparaciones_pareadas.py results/fold_level_results.csv out/
python analysis/analisis_fase2.py results/zeroshot out/
python analysis/analisis_ddi_fototipo.py ddi_metadata.csv out/ddi.csv \
    MobileNetV2=results/zeroshot/FINAL_C40_mobilenetv2/DDI/predicciones_zeroshot_DDI.csv:0.639 \
    EfficientNetB0=results/zeroshot/FINAL_C40_efficientnetb0/DDI/predicciones_zeroshot_DDI.csv:0.629
```

The thresholds passed to `analisis_ddi_fototipo.py` are the frozen thresholds stored in each `resumen_zeroshot_DDI.json`.

## 4. Figures (`figures/`)

| Script | Figure |
|---|---|
| `figura_metodologia.py out/methodology` | experimental framework (PDF + PNG) |
| `figura_roc_internas.py results/predictions_cv/mobilenetv2 results/predictions_cv/efficientnetb0 out/roc.png out/auc.csv` | out-of-fold ROC curves on Brown Skin (repeats 2 and 3) |
| `figura_roc_zeroshot.py results/zeroshot/FINAL_C40_mobilenetv2 results/zeroshot/FINAL_C40_efficientnetb0 out/roc_zs.png` | zero-shot ROC curves |
| `figura_A_vs_C40.py out/auc_fase2.csv out/fig.png` | A vs. B vs. C40 external AUC (input: output of `analisis_fase2.py`) |
| `figura_baseline_auc.py`, `figura_c40_targets.py` | single-domain baseline and C40-target bar charts (values written in the script) |

## Results included (`results/`)

- `fold_level_results.csv`: one row per configuration, backbone, repeat and fold (lesion-level partition).
- `predictions_cv/`: raw out-of-fold scores of every cross-validation run.
- `zeroshot/FINAL_<config>_<backbone>[_s<seed>]/<cohort>/`: per-image zero-shot scores and summaries of the ten final models.
- `control_brownskin/`: scores of the final C40 models on the Brown Skin hold-out.

## Notes on reproducibility

- Under the lesion-level partition, MobileNetV2 A, B and C40 were run with an earlier revision of the training script than C10–C80, D and E, with the same partitions and seeds; EfficientNetB0 and all final models were run with the current pipeline. The two prediction files of MobileNetV2 repeat 1, fold 1 (A and C40) do not match the recorded fold AUCs, so ROC figures use repeats 2 and 3.
- The image-level sensitivity analysis, the single-domain baseline (Strand 1) and the C40 runs with Fitzpatrick17k and PAD-UFES-20 as fine-tuning targets were produced with the earlier image-level pipeline and are not reproduced by these scripts.
- Two DDI-2 benign images cannot be decoded and are skipped (661 of 663 images).

## Model weights

The trained weights of the ten final models (`FINAL_*.keras`) are archived on Zenodo: https://doi.org/10.5281/zenodo.23743997 

## Citation

*[add after publication]*
