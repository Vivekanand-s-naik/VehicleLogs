@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM ANPR Gate System — Protected Standalone Windows Executable Builder (Nuitka)
REM ============================================================================
REM Compiles Python source (run.py, app/, inference_onnx.py, license_check.py)
REM into native C-compiled binaries without shipping .py source files, and
REM bundles the FP32 .onnx models, static web UI, and LICENSE_NOTICE.txt.
REM
REM Prerequisites:
REM   - Python 3.10 or 3.11 (64-bit) on Windows
REM   - Internet access on first run (to install minimal runtime packages & Nuitka)
REM
REM Optional environment overrides:
REM   set PYTHON_EXE=C:\Path\To\python.exe   (base Python to create build venv)
REM   set BUILD_VENV=.venv-runtime-build     (isolated clean build environment)
REM ============================================================================

cd /d "%~dp0"

if "%BUILD_VENV%"=="" set "BUILD_VENV=.venv-runtime-build"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"

echo.
echo ============================================================================
echo  [1/6] Verifying required ONNX models and disclosure files...
echo ============================================================================

if not exist "models\license_plate_detector.onnx" (
    echo [ERROR] Missing models\license_plate_detector.onnx
    echo         Run "python export_to_onnx.py" first to export and verify models.
    exit /b 1
)
if not exist "models\vehicle_detector.onnx" (
    echo [ERROR] Missing models\vehicle_detector.onnx
    echo         Run "python export_to_onnx.py" first to export and verify models.
    exit /b 1
)
if not exist "LICENSE_NOTICE.txt" (
    echo [ERROR] Missing LICENSE_NOTICE.txt disclosure file.
    exit /b 1
)
if not exist "app\static\index.html" (
    echo [ERROR] Missing app\static\index.html web assets.
    exit /b 1
)

echo   OK: models\license_plate_detector.onnx
echo   OK: models\vehicle_detector.onnx
echo   OK: LICENSE_NOTICE.txt
echo   OK: app\static\

echo.
echo ============================================================================
echo  [2/6] Preparing clean minimal runtime build environment (%BUILD_VENV%)...
echo ============================================================================

if not exist "%BUILD_VENV%\Scripts\python.exe" (
    echo Creating isolated build virtual environment...
    "%PYTHON_EXE%" -m venv "%BUILD_VENV%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment using %PYTHON_EXE%.
        exit /b 1
    )
)

set "VENV_PY=%CD%\%BUILD_VENV%\Scripts\python.exe"
set "VENV_PIP=%CD%\%BUILD_VENV%\Scripts\pip.exe"

echo Upgrading pip and installing minimal runtime dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

"%VENV_PIP%" install -r requirements-runtime.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements-runtime.txt
    exit /b 1
)

echo Ensuring onnxruntime-directml (DmlExecutionProvider) is not shadowed by CPU-only onnxruntime...
"%VENV_PIP%" uninstall -y onnxruntime >nul 2>&1
"%VENV_PIP%" install --no-deps --force-reinstall "onnxruntime-directml>=1.17.0"
if errorlevel 1 (
    echo [ERROR] Failed to configure onnxruntime-directml
    exit /b 1
)

echo Installing Nuitka build toolchain (build-time only, not shipped to users)...
"%VENV_PIP%" install --upgrade nuitka ordered-set zstandard
if errorlevel 1 (
    echo [ERROR] Failed to install Nuitka
    exit /b 1
)

echo.
echo ============================================================================
echo  [3/6] Verifying DirectML + CPU ONNX Runtime providers in build venv...
echo ============================================================================
"%VENV_PY%" -c "import onnxruntime as ort; from inference_onnx import select_onnx_providers; print('Available:', ort.get_available_providers()); print('Selected :', select_onnx_providers()); assert 'CPUExecutionProvider' in ort.get_available_providers()"
if errorlevel 1 (
    echo [ERROR] ONNX Runtime verification failed inside %BUILD_VENV%.
    exit /b 1
)

echo.
echo ============================================================================
echo  [4/6] Compiling standalone protected Windows executable with Nuitka...
echo        Flags: --standalone --windows-console-mode=disable
echo ============================================================================

if exist "dist\run.dist" rmdir /s /q "dist\run.dist"
if exist "dist\run.build" rmdir /s /q "dist\run.build"

"%VENV_PY%" -m nuitka ^
    --standalone ^
    --windows-console-mode=disable ^
    --assume-yes-for-downloads ^
    --output-dir=dist ^
    --output-filename=ANPR_GateSystem.exe ^
    --product-name="ANPR Gate Access System" ^
    --file-version=1.0.0.0 ^
    --product-version=1.0.0.0 ^
    --include-package=app ^
    --include-module=inference_onnx ^
    --include-module=license_check ^
    --include-package=uvicorn.logging ^
    --include-package=uvicorn.loops ^
    --include-package=uvicorn.protocols ^
    --include-package=uvicorn.lifespan ^
    --include-package-data=onnxruntime ^
    --include-package-data=rapidocr_onnxruntime ^
    --include-package-data=fast_plate_ocr ^
    --include-data-file=models/license_plate_detector.onnx=models/license_plate_detector.onnx ^
    --include-data-file=models/vehicle_detector.onnx=models/vehicle_detector.onnx ^
    --include-data-dir=app/static=app/static ^
    --include-data-file=LICENSE_NOTICE.txt=LICENSE_NOTICE.txt ^
    --include-data-file=.env.example=.env.example ^
    --nofollow-import-to=torch ^
    --nofollow-import-to=torchvision ^
    --nofollow-import-to=torchaudio ^
    --nofollow-import-to=ultralytics ^
    --nofollow-import-to=easyocr ^
    --nofollow-import-to=onnx ^
    --nofollow-import-to=pytest ^
    --nofollow-import-to=streamlit ^
    --nofollow-import-to=matplotlib ^
    --nofollow-import-to=pandas ^
    --nofollow-import-to=scipy ^
    --nofollow-import-to=IPython ^
    --nofollow-import-to=notebook ^
    --nofollow-import-to=tkinter ^
    run.py

if errorlevel 1 (
    echo.
    echo [ERROR] Nuitka compilation failed. Check compiler output above.
    exit /b 1
)

echo.
echo ============================================================================
echo  [5/6] Writing packaged runtime .env template (ONNX + License Check ON)...
echo ============================================================================

set "DIST_DIR=dist\run.dist"
if not exist "%DIST_DIR%\ANPR_GateSystem.exe" (
    echo [ERROR] Expected executable not found at %DIST_DIR%\ANPR_GateSystem.exe
    exit /b 1
)

(
echo # Packaged Windows Runtime Configuration (ONNX Runtime DirectML/CPU)
echo INFERENCE_BACKEND=onnx
echo DEVICE=auto
echo CAMERA_SOURCE=0
echo PLATE_ONNX_MODEL_PATH=models/license_plate_detector.onnx
echo VEHICLE_ONNX_MODEL_PATH=models/vehicle_detector.onnx
echo API_HOST=0.0.0.0
echo API_PORT=8000
echo.
echo # Disclosed Licensing ^(see LICENSE_NOTICE.txt^)
echo LICENSE_CHECK_ENABLED=true
echo LICENSE_SERVER_URL=https://my-license-server.example.com/api/v1/validate
echo LICENSE_KEY=REPLACE_WITH_CUSTOMER_LICENSE_KEY
echo LICENSE_SUPPORT_CONTACT=support@yourcompany.example.com
echo LICENSE_GRACE_HOURS=72
echo LICENSE_TIMEOUT_SECONDS=6.0
echo LICENSE_MAX_RETRIES=3
echo LICENSE_RECHECK_INTERVAL_SECONDS=3600
) > "%DIST_DIR%\.env"

echo.
echo ============================================================================
echo  [6/6] Build complete!
echo ============================================================================
echo   Standalone Bundle : %CD%\%DIST_DIR%\
echo   Executable        : %CD%\%DIST_DIR%\ANPR_GateSystem.exe
echo   Bundled Models    : %CD%\%DIST_DIR%\models\*.onnx
echo   Disclosure Notice : %CD%\%DIST_DIR%\LICENSE_NOTICE.txt
echo   Runtime Config    : %CD%\%DIST_DIR%\.env (update LICENSE_SERVER_URL ^& LICENSE_KEY)
echo.
echo   NOTE: Original .pt weights and .py source files are NOT copied to dist\.
echo ============================================================================
exit /b 0
