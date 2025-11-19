@echo off
REM End-to-end pipeline runner for Windows
REM Captures all output to pipeline_run.log

set SCRIPT_DIR=scripts
set LOG_FILE=pipeline_run.log

echo [run_all] Starting Earth Observation pipeline...
echo [run_all] Output will be logged to %LOG_FILE%
echo.

REM Clear previous log
if exist %LOG_FILE% del %LOG_FILE%

REM Step 1: Create metadata
echo [run_all] Step 1/6: Creating metadata...
python %SCRIPT_DIR%\create_metadata.py --rgb-dir Data\rgb --output Dataset\metadata.csv >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo [run_all][ERROR] Step 1 failed - check %LOG_FILE%
    exit /b 1
)

REM Step 2: Label from landcover
echo [run_all] Step 2/6: Labeling from landcover...
python %SCRIPT_DIR%\label_from_landcover.py --metadata Dataset\metadata.csv --tif Data\worldcover_bbox_delhi_ncr_2021.tif --patch-size 128 --dominance-threshold 0.0 --threads 4 --output Dataset\labels.csv >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo [run_all][ERROR] Step 2 failed - check %LOG_FILE%
    exit /b 1
)

REM Step 3: Split dataset
echo [run_all] Step 3/6: Splitting dataset...
python %SCRIPT_DIR%\split_dataset.py --metadata Dataset\metadata.csv --labels Dataset\labels.csv --test-size 0.4 --random-state 42 >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo [run_all][ERROR] Step 3 failed - check %LOG_FILE%
    exit /b 1
)

REM Step 4: Train model
echo [run_all] Step 4/6: Training model...
python %SCRIPT_DIR%\train_model.py --model cnn --epochs 10 --batch-size 32 --img-dir Data\rgb --lr 1e-3 --use-cuda --class-weight >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo [run_all][ERROR] Step 4 failed - check %LOG_FILE%
    exit /b 1
)

REM Step 5: Visualize results
echo [run_all] Step 5/6: Visualizing results...
python %SCRIPT_DIR%\visualize_results.py --predictions runs\predictions.csv --img-dir Data\rgb --n-correct 5 --n-incorrect 5 --out-dir Visualizations >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo [run_all][ERROR] Step 5 failed - check %LOG_FILE%
    exit /b 1
)

REM Step 6: Grid map (optional, continue on error)
echo [run_all] Step 6/6: Creating grid map (optional)...
python %SCRIPT_DIR%\grid_map.py --metadata Dataset\metadata.csv --html-out Visualizations\grid_map.html --png-out Visualizations\grid_map.png >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo [run_all][WARNING] Step 6 failed (non-critical) - check %LOG_FILE%
)

REM Generate report
echo [run_all] Generating PDF report...
python %SCRIPT_DIR%\report_builder.py --mode final --report report.pdf >> %LOG_FILE% 2>&1
if errorlevel 1 (
    echo [run_all][WARNING] Report generation failed (non-critical) - check %LOG_FILE%
)

REM Extract summary from log
echo.
echo [run_all] ========================================
echo [run_all] PIPELINE COMPLETE
echo [run_all] ========================================
echo.
echo Summary (check %LOG_FILE% for details):
findstr /C:"images scanned" /C:"labels created" /C:"Train:" /C:"Test:" /C:"Final Macro F1" /C:"Best model saved" /C:"PDF report saved" %LOG_FILE%
echo.
echo Full log: %LOG_FILE%
