@echo off
REM ==========================================================================
REM  Fase 4a (28-oct-2026): sin entrenar nada.
REM   1) Zero-shot de los 18 modelos finales en Derm7pt (misma lesion en
REM      dermatoscopia y en foto clinica) + busqueda de duplicados con HAM10000.
REM   2) Tono de piel (ITA) de todas las bases.
REM  Ejecutar desde C:\1. HAM10000 (donde esta entrenar_leakage_safe_v4.py).
REM  DERM7PT: carpeta donde descomprimiste Derm7pt (el script busca meta.csv dentro)
REM ==========================================================================
setlocal
cd /d "%~dp0"
set DERM7PT=C:\1. HAM10000\derm7pt

echo [1/2] Derm7pt
python evaluar_derm7pt.py --derm7pt_dir "%DERM7PT%" --ham_dir "datos_piel\pieles_claras" || goto :error
echo [2/2] ITA
python medir_ita.py || goto :error
powershell -NoProfile -Command "Compress-Archive -Force -Path 'resultados\leakage_safe\derm7pt','resultados\leakage_safe\ita' -DestinationPath 'resultados_fase4a.zip'"
echo.
echo LISTO. Envia resultados_fase4a.zip
pause
exit /b 0
:error
echo *** Se detuvo por un error. Copia el mensaje y envialo. ***
pause
exit /b 1
