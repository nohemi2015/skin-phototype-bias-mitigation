@echo off
REM ==========================================================================
REM  Fase 5 (29-oct-2026): modelos finales de FUSION (D: HAM10000 + Brown Skin,
REM  y E: fusion balanceada), 3 semillas x 2 backbones = 12 modelos, con el mismo
REM  protocolo que A, B y C40. Despues: zero-shot en las 4 cohortes y en Derm7pt
REM  (salta los modelos ya evaluados) y verificacion de los modelos finales.
REM  Pregunta: usar Brown Skin JUNTO con HAM10000 conserva el desempeno en
REM  imagenes reales y mejora en piel oscura?
REM  Ejecutar desde C:\1. HAM10000. Si se interrumpe, comenta con REM los
REM  entrenamientos ya terminados y vuelve a lanzarlo.
REM ==========================================================================
setlocal
cd /d "%~dp0"
set DERM7PT=C:\1. HAM10000\derm7pt
for %%M in (D E) do (
  for %%R in (mobilenetv2 efficientnetb0) do (
    for %%S in (2026 2027 2028) do (
      echo === %%M %%R semilla %%S ===
      python entrenar_leakage_safe_v4.py --model %%M --train_final --arch %%R --final_seed %%S || goto :error
    )
  )
)
python correr_zeroshot_lote.py || goto :error
python evaluar_derm7pt.py --derm7pt_dir "%DERM7PT%" || goto :error
python verificar_modelos_finales.py || goto :error
powershell -NoProfile -Command "Compress-Archive -Force -Path 'resultados\leakage_safe\zeroshot_lote','resultados\leakage_safe\derm7pt\predicciones_derm7pt.csv','verificacion_modelos.zip' -DestinationPath 'resultados_fase5.zip'"
echo.
echo LISTO. Envia resultados_fase5.zip
pause
exit /b 0
:error
echo *** Se detuvo por un error. Copia el mensaje y envialo. ***
pause
exit /b 1
