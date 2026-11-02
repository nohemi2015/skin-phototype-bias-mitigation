@echo off
REM ==========================================================================
REM  Fase 4b (28-oct-2026): repite la validacion cruzada de MobileNetV2 con el
REM  script actual (A, B, C40 y despues C10, C20, C60, C80), en una carpeta
REM  NUEVA (rerun_cv) para no tocar ni saltar los resultados anteriores.
REM  Asi todas las configuraciones usan la misma version del codigo y el mismo
REM  modelo A como punto de partida, y se regeneran los archivos de prediccion.
REM  Se puede interrumpir y volver a lanzar: salta los folds ya terminados.
REM  Ejecutar desde C:\1. HAM10000
REM ==========================================================================
setlocal
cd /d "%~dp0"
set LS_OUT=%~dp0rerun_cv
for %%M in (A B C40 C10 C20 C60 C80) do (
  echo === %%M MobileNetV2 ===
  python entrenar_leakage_safe_v4.py --model %%M --arch mobilenetv2 || goto :error
)
powershell -NoProfile -Command "Compress-Archive -Force -Path 'rerun_cv\resultados\leakage_safe\resultados_leakage_safe.csv','rerun_cv\resultados\leakage_safe\predicciones','rerun_cv\resultados\leakage_safe\split_ledger.csv' -DestinationPath 'resultados_fase4b.zip'"
echo.
echo LISTO. Envia resultados_fase4b.zip
pause
exit /b 0
:error
echo *** Se detuvo por un error. Copia el mensaje y envialo. ***
pause
exit /b 1
