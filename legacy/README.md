# Legacy scripts

Earlier scripts kept for transparency. They were used only for analyses reported in the Supplementary Material or as descriptive baselines:

- `entrenar_ham10000_modelA_*.py`, `entrenar_ddi2_modelA*.py`, `entrenar_ham_ddi.py`: single-domain baseline (Strand 1).
- `entrenar_ham_fitzpatrick.py`, `entrenar_efficiennetb0_fitzpatrick.py`, `entrenar_ham_padufes*.py`: C40 adaptation toward real targets (Supplementary Section S2, image-level split).
- `entrenar_leakage_safe_v2.py`, `entrenar_groupkfold_bs.py`, `evaluar_zeroshot*.py`: earlier versions of the current pipeline; image-level sensitivity analysis (Supplementary Section S1).
- `config.py`, `dataset.py`, `models.py`, `evaluation.py`, `clasificar_benignas_malignas.py`, `generar_roc_*.py`, `graficar_roc_*.py`: helpers and plotting scripts of the earlier pipeline.

None of these scripts produces a primary result of the article; the primary analyses are in `pipeline/`, `analysis/` and `figures/`.
