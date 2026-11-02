# Domain adaptation to synthetic dark-skin dermoscopic images: lesion-level and zero-shot external evaluation

Code, per-image predictions and fold-level results for the article

> N. Sánchez-Medel, R. Díaz-Hernández, E. Ruiz-Muñoz, S. Zapotecas-Martínez, L. Altamirano-Robles, J. J. Velázquez-Arreola.
> *Evaluating Domain Adaptation on Synthetic Dark-Skin Dermoscopic Images: Internal Gains Did Not Transfer to Real Clinical Photographs* (under review).

The study adapts MobileNetV2 and EfficientNetB0 from HAM10000 to the style-transfer-derived Brown Skin dataset, evaluates every configuration under a lesion-level cross-validation that keeps a lesion and all its synthetic derivatives in the same fold, and then evaluates frozen final models zero-shot on four cohorts of real clinical photographs (DDI, DDI-2, Fitzpatrick17k, PAD-UFES-20) and on Derm7pt, whose paired dermoscopic and clinical images of the same lesions serve as a modality control.

## Repository layout

```
pipeline/    training, final models and zero-shot evaluation
analysis/    statistics reported in the article
figures/     scripts that draw the figures
results/     fold-level results and per-image predictions used in the article
legacy/      earlier scripts used only for the single-domain baseline, the image-level
             sensitivity analysis and the real-target C40 runs (Supplementary Material)
```

## Split ledger

`results/split_ledger.zip` contains `split_ledger.csv` (zipped because it exceeds GitHub's upload limit): for every image and every fold of the three repeats: dataset of origin, repeat, fold, partition (train or validation), file path (the file name contains the HAM10000 `image_id`), `lesion_id` and label.

## Requirements

Python 3.10+ and the packages in `requirements.txt` (TensorFlow/Keras, scikit-learn, pandas, numpy, scipy, matplotlib). All models were trained on CPU.

## Data

Images are **not** redistributed. Download them from their official sources:

| Dataset | Role | Source |
|---|---|---|
| HAM10000 (+ `HAM10000_metadata.csv`) | source domain | Tschandl et al., *Sci. Data* 2018; images used here were downloaded from the ISIC Archive |
| Brown Skin | synthetic target | Sánchez-Medel et al., *Computación y Sistemas* 2026; available from the corresponding author on reasonable request, or regenerated from HAM10000 with the published procedure |
| DDI (+ `ddi_metadata.csv`) | external | Daneshjou et al., *Sci. Adv.* 2022 (Stanford AIMI) |
| DDI-2 | external | Chang et al., *J. Invest. Dermatol.* 2024 |
| Fitzpatrick17k (FST IV–VI, benign/malignant) | external | Groh et al., CVPRW 2021 |
| PAD-UFES-20 | external | Pacheco et al., *Data in Brief* 2020 |

The pipeline expects this layout next to the scripts (each class folder holds the images of that class):

```
HAM10000_metadata.csv
datos_piel/
  pieles_claras/{BENIGNAS,MALIGNAS}/          HAM10000 (9,672 images; see results/verification/ham10000_excluidas.csv)
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

`--train_final` trains one model on a lesion-level 85 % development split (the same split for every model and seed), freezes the Youden threshold on Brown Skin validation images, and evaluates the Brown Skin images of the 15 % hold-out as a same-distribution control. `correr_zeroshot_lote.py` evaluates every `FINAL_*.json` on the four cohorts (AUC, sensitivity and specificity at the frozen and at a re-optimized threshold, bootstrap CIs). `correr_fase2.bat` and `correr_semillas_AB.bat` run the full set used in the article on Windows: A, B and C40 with seeds 2026, 2027 and 2028 for both backbones; `correr_fase5_fusion.bat` adds the fusion models D and E (E is not reported; see the note below). The article uses 24 final models. `verificar_modelos_finales.py` reloads every final model, re-predicts the hold-out and compares it with the archived predictions, and summarizes the training histories (`results/verification/`).

## 3. Statistics (`analysis/`)

| Script | Input | Output in the article |
|---|---|---|
| `comparaciones_pareadas.py` | `results/fold_level_results.csv` | MobileNetV2 lesion-level summary and paired comparisons (corrected repeated-CV *t*-test, Holm) |
| `diferencias_por_repeticion.py` | `results/fold_level_results.csv` | per-repeat sensitivity analysis of those comparisons |
| `analisis_arquitecturas.py` | `results/fold_level_results.csv` | both-backbone summary; EfficientNetB0 and between-backbone paired comparisons |
| `analisis_zeroshot_dos_modelos.py` | `results/zeroshot/FINAL_C40_mobilenetv2`, `.../FINAL_C40_efficientnetb0` | zero-shot CIs (patients resampled for PAD-UFES-20) and EfficientNetB0 − MobileNetV2 differences |
| `analisis_fase2.py` | `results/zeroshot/`, `results/control_brownskin/` | A, B, C40, D and E (3 seeds each; E not reported) external and control AUCs; paired differences of the seed means (A − C40, B − C40, A − B, A − D, A − E; Holm over 8 comparisons per family) |
| `analisis_derm7pt.py` | `results/derm7pt/predicciones_derm7pt.csv`, `results/verification/verificacion_modelos_finales.csv` | Derm7pt dermoscopic vs. clinical AUC of all final models |
| `analisis_ddi_fototipo.py` | `ddi_metadata.csv` + DDI predictions | DDI AUC by Fitzpatrick group, all positives and MEL/BCC/SCC only |

Examples:

```
python analysis/comparaciones_pareadas.py results/fold_level_results.csv out/
python analysis/analisis_fase2.py results/zeroshot results/control_brownskin out/
python analysis/analisis_ddi_fototipo.py ddi_metadata.csv out/ddi.csv \
    MobileNetV2=results/zeroshot/FINAL_C40_mobilenetv2/DDI/predicciones_zeroshot_DDI.csv:0.639 \
    EfficientNetB0=results/zeroshot/FINAL_C40_efficientnetb0/DDI/predicciones_zeroshot_DDI.csv:0.629
```

The thresholds passed to `analisis_ddi_fototipo.py` are the frozen thresholds stored in each `resumen_zeroshot_DDI.json`.

## 4. Figures (`figures/`)

| Script | Figure |
|---|---|
| `figura_metodologia.py out/methodology` | experimental framework (PDF + PNG) |
| `figura_roc_internas.py results/predictions_cv/mobilenetv2 results/predictions_cv/efficientnetb0 out/roc.png out/auc.csv` | out-of-fold ROC curves on Brown Skin (all three repeats) |
| `figura_roc_zeroshot.py results/zeroshot/FINAL_C40_mobilenetv2 results/zeroshot/FINAL_C40_efficientnetb0 out/roc_zs.png` | zero-shot ROC curves |
| `figura_A_vs_C40.py out/auc_fase2.csv out/fig.png` | A vs. B vs. C40 external AUC (input: output of `analisis_fase2.py`) |
| `figura_resumen_grafico.py out/auc_fase2.csv out/graphical_abstract` | graphical abstract (PDF, PNG, TIFF) |
| `figura_derm7pt_ita.py out/auc_derm7pt.csv results/ita/ita_por_imagen.csv out/fig.png` | Derm7pt modality control and skin-tone (ITA) figure |
| `figura_baseline_auc.py`, `figura_c40_targets.py` | single-domain baseline and C40-target bar charts (values written in the script) |

## Results included (`results/`)

- `fold_level_results.csv`: one row per configuration, backbone, repeat and fold (lesion-level partition).
- `predictions_cv/`: raw out-of-fold scores of every cross-validation run.
- `zeroshot/FINAL_<config>_<backbone>[_s<seed>]/<cohort>/`: per-image zero-shot scores and summaries of all final models.
- `control_brownskin/`: scores of all final models on the Brown Skin hold-out (with `lesion_id`).
- `verification/`: audit of the archived predictions (`cv_prediction_audit.csv`: fold AUC recomputed from every cross-validation prediction file versus the recorded value) and of the final models.

## Notes on reproducibility

- All lesion-level configurations of both backbones were run with the current training script (MobileNetV2 A, B and C10–C80 were re-run with the same partitions and seeds; the regenerated split ledger is identical). Every archived cross-validation prediction file reproduces its recorded fold AUC, and the ROC figures use all three repeats.
- The image-level sensitivity analysis, the single-domain baseline (Strand 1) and the C40 runs with Fitzpatrick17k and PAD-UFES-20 as fine-tuning targets were produced with the earlier image-level pipeline and are not reproduced by these scripts.
- 343 HAM10000 images (all BKL, all Brown Skin parents) were not included in `pieles_claras`; the files are intact in the ISIC download. Their `image_id`s are in `results/verification/ham10000_excluidas.csv`, and `pipeline/buscar_ham_faltantes.py` locates them in a local copy.
- Two DDI-2 benign images cannot be decoded and are skipped (661 of 663 images).

## Note on strategy E (not reported in the article)

Strategy E ("balanced fusion") was meant to oversample Brown Skin to balance the two domains. Its replication factor is `len(HAM train) // len(BS train)` with a minimum of 1; because the Brown Skin training set is larger than the HAM10000 training set (about 9,400 vs. 7,700 images per fold), the factor is 1 in every fold and in every final model, so E trains on exactly the same images as D, in a different order. Its results reproduce D (internal AUC within 0.001; external and Derm7pt AUC within 0.01), and it is therefore not reported in the article. Its fold-level results and predictions, and the earlier analysis scripts that include it (`*_conE.py`), are kept for completeness. The article reports 24 final models (A, B, C40 and D, two backbones, three seeds).

## Model weights

The trained weights of the final models (`FINAL_*.keras`) are archived on Zenodo: https://doi.org/10.5281/zenodo.23743997

## Citation

*[add after publication]*
