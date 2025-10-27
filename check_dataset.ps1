# Dataset Diagnostic Script for YOLO-UDD v2.0
# Run this to check your TrashCan dataset setup

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  YOLO-UDD v2.0 - Dataset Diagnostic Tool" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

$datasetPath = "F:\MIR\project\YOLO-UDD-v2.0-main\data\trashcan"

Write-Host "Checking dataset location: $datasetPath" -ForegroundColor Yellow

# Check if directory exists
if (Test-Path $datasetPath) {
    Write-Host "  [OK] Directory exists`n" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Directory not found!`n" -ForegroundColor Red
    exit 1
}

# Check annotation files
Write-Host "Checking annotation files:" -ForegroundColor Yellow
$trainJson = Join-Path $datasetPath "instances_train_trashcan.json"
$valJson = Join-Path $datasetPath "instances_val_trashcan.json"

if (Test-Path $trainJson) {
    $size = (Get-Item $trainJson).Length
    Write-Host "  [OK] instances_train_trashcan.json ($size bytes)" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] instances_train_trashcan.json" -ForegroundColor Red
}

if (Test-Path $valJson) {
    $size = (Get-Item $valJson).Length
    Write-Host "  [OK] instances_val_trashcan.json ($size bytes)" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] instances_val_trashcan.json" -ForegroundColor Red
}

# Check images directories
Write-Host "`nChecking image directories:" -ForegroundColor Yellow
$trainImgDir = Join-Path $datasetPath "images\train"
$valImgDir = Join-Path $datasetPath "images\val"

if (Test-Path $trainImgDir) {
    if ((Get-Item $trainImgDir) -is [System.IO.DirectoryInfo]) {
        $trainImages = @(Get-ChildItem $trainImgDir -Filter *.jpg -File -ErrorAction SilentlyContinue)
        $trainImagesPng = @(Get-ChildItem $trainImgDir -Filter *.png -File -ErrorAction SilentlyContinue)
        $totalTrain = $trainImages.Count + $trainImagesPng.Count
        Write-Host "  [OK] images/train/ - $totalTrain images found" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] images/train exists but is not a directory!" -ForegroundColor Red
    }
} else {
    Write-Host "  [MISSING] images/train/ directory" -ForegroundColor Red
}

if (Test-Path $valImgDir) {
    if ((Get-Item $valImgDir) -is [System.IO.DirectoryInfo]) {
        $valImages = @(Get-ChildItem $valImgDir -Filter *.jpg -File -ErrorAction SilentlyContinue)
        $valImagesPng = @(Get-ChildItem $valImgDir -Filter *.png -File -ErrorAction SilentlyContinue)
        $totalVal = $valImages.Count + $valImagesPng.Count
        Write-Host "  [OK] images/val/ - $totalVal images found" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] images/val exists but is not a directory!" -ForegroundColor Red
    }
} else {
    Write-Host "  [MISSING] images/val/ directory" -ForegroundColor Red
}

# Summary
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

$issues = 0

if (!(Test-Path $trainJson)) { $issues++ }
if (!(Test-Path $valJson)) { $issues++ }
if (!(Test-Path $trainImgDir)) { $issues++ }
if (!(Test-Path $valImgDir)) { $issues++ }

if ($issues -eq 0 -and $totalTrain -gt 0 -and $totalVal -gt 0) {
    Write-Host "  STATUS: READY" -ForegroundColor Green
    Write-Host "  Your dataset appears to be properly set up!`n" -ForegroundColor Green
} else {
    Write-Host "  STATUS: INCOMPLETE ($issues issues found)" -ForegroundColor Red
    Write-Host "`n  Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Review DATASET_FIX_GUIDE.md for detailed instructions" -ForegroundColor White
    Write-Host "  2. Download the real TrashCan dataset, OR" -ForegroundColor White
    Write-Host "  3. Generate dummy dataset: python scripts/create_dummy_dataset.py`n" -ForegroundColor White
}

# Check Python installation
Write-Host "Checking Python installation:" -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  [OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [WARNING] Python not found in PATH" -ForegroundColor Yellow
    Write-Host "            Install from: https://www.python.org/downloads/`n" -ForegroundColor White
}

Write-Host "`nFor more help, see:" -ForegroundColor Cyan
Write-Host "  - DATASET_FIX_GUIDE.md (complete troubleshooting guide)" -ForegroundColor White
Write-Host "  - DATASET_SETUP_INSTRUCTIONS.md (download instructions)" -ForegroundColor White
Write-Host "  - DOCUMENTATION.md (project overview)`n" -ForegroundColor White
