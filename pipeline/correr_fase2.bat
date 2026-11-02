@echo off
REM ==========================================================================
REM  Fase 2 (27-oct-2026): modelos finales A, B y C40 con 2 semillas extra,
REM  y evaluacion zero-shot de todos en las 4 cohortes.
REM  Copiar a la carpeta del proyecto (donde esta entrenar_leakage_safe_v3.py)
REM  y ejecutar con doble clic o desde la consola: correr_fase2.bat
REM  Si se interrumpe, se puede volver a lanzar: el zero-shot salta lo ya hecho
REM  (los entrenamientos si se repiten; comentar con REM los que ya terminaron).
REM ==========================================================================
cd /d "%~dp0"

echo [1/9] A final - MobileNetV2
python entrenar_leakage_safe_v4.py --model A --train_final --arch mobilenetv2 || goto :error
echo [2/9] A final - EfficientNetB0
python entrenar_leakage_safe_v4.py --model A --train_final --arch efficientnetb0 || goto :error
echo [3/9] B final - MobileNetV2
python entrenar_leakage_safe_v4.py --model B --train_final --arch mobilenetv2 || goto :error
echo [4/9] B final - EfficientNetB0
python entrenar_leakage_safe_v4.py --model B --train_final --arch efficientnetb0 || goto :error
echo [5/9] C40 final semilla 2027 - MobileNetV2
python entrenar_leakage_safe_v4.py --model C40 --train_final --arch mobilenetv2 --final_seed 2027 || goto :error
echo [6/9] C40 final semilla 2028 - MobileNetV2
python entrenar_leakage_safe_v4.py --model C40 --train_final --arch mobilenetv2 --final_seed 2028 || goto :error
echo [7/9] C40 final semilla 2027 - EfficientNetB0
python entrenar_leakage_safe_v4.py --model C40 --train_final --arch efficientnetb0 --final_seed 2027 || goto :error
echo [8/9] C40 final semilla 2028 - EfficientNetB0
python entrenar_leakage_safe_v4.py --model C40 --train_final --arch efficientnetb0 --final_seed 2028 || goto :error

echo [9/9] Zero-shot de todos los modelos finales
python correr_zeroshot_lote.py || goto :error

echo Empaquetando resultados...
powershell -NoProfile -Command "Compress-Archive -Force -Path 'resultados\leakage_safe\zeroshot_lote','resultados\leakage_safe\FINAL_*.json','resultados\leakage_safe\predicciones\pred_FINAL_*' -DestinationPath 'resultados_fase2.zip'"
echo.
echo LISTO. Envia el archivo resultados_fase2.zip
pause
exit /b 0

:error
echo.
echo *** Se detuvo por un error. Copia el mensaje de arriba y envialo. ***
pause
exit /b 1
