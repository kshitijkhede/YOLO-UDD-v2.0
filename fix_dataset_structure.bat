@echo off
REM Quick fix script for YOLO-UDD v2.0 dataset structure

echo.
echo ============================================================
echo   YOLO-UDD v2.0 - Quick Dataset Structure Fix
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking current structure...
if exist "data\trashcan\images\train" (
    echo   Found: data\trashcan\images\train
) else (
    echo   Missing: data\trashcan\images\train
)

if exist "data\trashcan\images\val" (
    echo   Found: data\trashcan\images\val
) else (
    echo   Missing: data\trashcan\images\val
)

echo.
echo [2/3] Fixing directory structure...

REM Remove files if they exist
if exist "data\trashcan\images\train" (
    del /F /Q "data\trashcan\images\train" 2>nul
    echo   Removed train file
)

if exist "data\trashcan\images\val" (
    del /F /Q "data\trashcan\images\val" 2>nul
    echo   Removed val file
)

REM Create directories
if not exist "data\trashcan\images\train" (
    mkdir "data\trashcan\images\train"
    echo   Created: data\trashcan\images\train\
)

if not exist "data\trashcan\images\val" (
    mkdir "data\trashcan\images\val"
    echo   Created: data\trashcan\images\val\
)

echo.
echo [3/3] Verifying structure...

if exist "data\trashcan\images\train\*" (
    echo   [OK] train directory is ready
) else (
    echo   [OK] train directory created (empty)
)

if exist "data\trashcan\images\val\*" (
    echo   [OK] val directory is ready
) else (
    echo   [OK] val directory created (empty)
)

echo.
echo ============================================================
echo   Structure Fix Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Add TrashCan dataset files to the directories above
echo   2. Run: powershell -ExecutionPolicy Bypass -File check_dataset.ps1
echo   3. Read: DATASET_FIX_GUIDE.md for complete instructions
echo.
echo Where to get dataset:
echo   - Real dataset: https://conservancy.umn.edu/handle/11299/214865
echo   - Dummy dataset: python scripts/create_dummy_dataset.py
echo.

pause
