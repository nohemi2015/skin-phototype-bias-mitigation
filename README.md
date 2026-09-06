# Mitigating Skin Phototype Bias in Dermatological AI

Code and results for a study on transfer learning and layer-freezing strategies
to mitigate skin phototype bias in dermatological image classification,
including external zero-shot validation on four independent real-world cohorts.

## Data

HAM10000, Fitzpatrick17k, DDI, DDI-2, and PAD-UFES-20 are publicly available
from their respective official sources; this repository does not redistribute
raw images. See Methods section of the manuscript for citation and access details.

## Experimental design (three strands)

- **Strand 1 (baseline):** single-domain training/evaluation on each of 6 datasets.
- **Strand 2 (adaptation):** HAM10000-pretrained models fine-tuned toward each
  target domain (Brown Skin, Fitzpatrick17k, PAD-UFES-20, DDI-2) with progressive
  layer-freezing (C10–C80).
- **Strand 3 (zero-shot external validation):** the best-performing C40 model,
  frozen, evaluated without retraining on four independent real cohorts.

## Key finding

Internal adaptation to a synthetic dark-skin dataset (Brown Skin, AUC = 0.816)
does not generalize zero-shot to real dark-skin cohorts (AUC = 0.51–0.59).
See `resumen_zeroshot_*.json` for full results with bootstrap 95% CIs.

## Reproducing the results

See docstrings at the top of each script for exact commands and arguments.

## Citation

[Pendiente: agrega aquí la referencia completa una vez publicado]
