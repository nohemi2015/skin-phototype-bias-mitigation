@echo off
setlocal
set LS_OUT=
cd /d "C:\1. HAM10000"
python reentrenar_c40_efficientnet.py --derm7pt_dir "C:\1. HAM10000\derm7pt"
powershell -Command "Compress-Archive -Path 'C:\1. HAM10000\resultados\leakage_safe\c40_eff_v2\*' -DestinationPath 'C:\1. HAM10000\c40_eff_v2.zip' -Force"
echo.
echo Listo: manda C:\1. HAM10000\c40_eff_v2.zip
pause
