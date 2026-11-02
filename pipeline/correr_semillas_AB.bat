@echo off
REM ==========================================================================
REM  Semillas 2027 y 2028 para los modelos finales A y B (ambos backbones),
REM  para que A, B y C40 tengan las mismas 3 semillas. Despues: zero-shot
REM  (salta lo ya hecho) y verificacion (junta los controles para enviar).
REM  Ejecutar desde C:\1. HAM10000
REM ==========================================================================
cd /d "%~dp0"
for %%S in (2027 2028) do (
  for %%M in (A B) do (
    for %%R in (mobilenetv2 efficientnetb0) do (
      echo === %%M %%R semilla %%S ===
      python entrenar_leakage_safe_v4.py --model %%M --train_final --arch %%R --final_seed %%S || goto :error
    )
  )
)
python correr_zeroshot_lote.py || goto :error
python verificar_modelos_finales.py || goto :error
powershell -NoProfile -Command "Compress-Archive -Force -Path 'resultados\leakage_safe\zeroshot_lote','verificacion_modelos.zip' -DestinationPath 'resultados_semillas_AB.zip'"
echo.
echo LISTO. Envia resultados_semillas_AB.zip
pause
exit /b 0
:error
echo *** Se detuvo por un error. Copia el mensaje y envialo. ***
pause
exit /b 1
