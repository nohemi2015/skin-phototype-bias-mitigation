@echo off
setlocal
cd /d "C:\1. HAM10000"
for %%S in ("" "_s2027" "_s2028") do (
  python cargar_c40_efficientnet.py --pesos "modelos\leakage_safe\c40_eff_v2\FINAL_C40_efficientnetb0%%~S_pesos.npz" --pred "resultados\leakage_safe\c40_eff_v2\pred_FINAL_C40_efficientnetb0%%~S_test_BS.csv"
)
pause
