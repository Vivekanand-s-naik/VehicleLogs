@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM ANPR Gate System — Protected Native Windows Desktop Builder (Nuitka + PyQt6)
REM ============================================================================
REM Compiles Python source (run.py, app/, app/gui/, inference_onnx.py,
REM license_check.py) into a standalone native PyQt6 Windows executable without
REM shipping .py source files or .pt weights, bundling ONNX Runtime DirectML
REM DLLs, FP32 .onnx models, Qt6 plugins, and LICENSE_NOTICE.txt.
REM
REM Prerequisites:
REM   - Python 3.10 or 3.11 (64-bit) on Windows
REM   - Internet access on first run (to install minimal runtime packages & Nuitka)
REM
REM Optional environment overrides:
REM   set PYTHON_EXE=C:\Path\To\python.exe   (base Python to create build venv)
REM   set BUILD_VENV=.venv-runtime-build     (isolated clean build environment)
REM   set NUITKA_JOBS=4                      (parallel C compiler workers)
REM ============================================================================

cd /d "%~dp0"

if "%BUILD_VENV%"=="" set "BUILD_VENV=.venv-runtime-build"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
if "%NUITKA_JOBS%"=="" set "NUITKA_JOBS=4"

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
if not exist "nuitka_onnx_dml.yml" (
    echo [ERROR] Missing nuitka_onnx_dml.yml Nuitka DirectML DLL configuration.
    exit /b 1
)

echo   OK: models\license_plate_detector.onnx
echo   OK: models\vehicle_detector.onnx
echo   OK: LICENSE_NOTICE.txt
echo   OK: nuitka_onnx_dml.yml

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
set "ORT_CAPI_DIR=%CD%\%BUILD_VENV%\Lib\site-packages\onnxruntime\capi"

set "NEED_PIP=0"
if "%FORCE_PIP%"=="1" set "NEED_PIP=1"
if not exist "%ORT_CAPI_DIR%\DirectML.dll" set "NEED_PIP=1"
"%VENV_PY%" -c "import PyQt6.QtWidgets, onnxruntime, cv2, numpy, rapidocr_onnxruntime, fast_plate_ocr, fastapi, uvicorn, pydantic_settings, sqlalchemy, jwt, bcrypt, requests, dotenv, nuitka" >nul 2>&1
if errorlevel 1 set "NEED_PIP=1"

if "%NEED_PIP%"=="1" (
    echo Installing minimal runtime dependencies ^(PyQt6 + ONNX Runtime DirectML^)...
    "%VENV_PIP%" install -r requirements-runtime.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements-runtime.txt
        exit /b 1
    )

    echo Ensuring single clean opencv-python build ^(removing conflicting opencv-python-headless^)...
    "%VENV_PIP%" show opencv-python-headless >nul 2>&1
    if not errorlevel 1 (
        "%VENV_PIP%" uninstall -y opencv-python-headless >nul 2>&1
        "%VENV_PIP%" install --no-deps --force-reinstall "opencv-python>=4.8.0,<4.12.0"
        if errorlevel 1 (
            echo [ERROR] Failed to configure clean opencv-python
            exit /b 1
        )
    )

    echo Ensuring onnxruntime-directml ^(DmlExecutionProvider^) is not shadowed by CPU-only onnxruntime...
    "%VENV_PIP%" show onnxruntime >nul 2>&1
    if not errorlevel 1 (
        "%VENV_PIP%" uninstall -y onnxruntime >nul 2>&1
        "%VENV_PIP%" install --no-deps --force-reinstall "onnxruntime-directml>=1.17.0"
        if errorlevel 1 (
            echo [ERROR] Failed to configure onnxruntime-directml
            exit /b 1
        )
    )

    "%VENV_PIP%" show nuitka >nul 2>&1
    if errorlevel 1 (
        "%VENV_PIP%" install --upgrade nuitka ordered-set zstandard
        if errorlevel 1 (
            echo [ERROR] Failed to install Nuitka
            exit /b 1
        )
    )
) else (
    echo All runtime dependencies ^(PyQt6, onnxruntime-directml, OCR, Nuitka^) already verified in %BUILD_VENV%.
)

if not exist "%ORT_CAPI_DIR%\DirectML.dll" (
    echo [ERROR] DirectML.dll not found in %ORT_CAPI_DIR%
    exit /b 1
)

echo.
echo ============================================================================
echo  [3/6] Verifying PyQt6 + DirectML/CPU ONNX Runtime in build venv...
echo ============================================================================
"%VENV_PY%" -c "import PyQt6.QtWidgets, onnxruntime as ort; from inference_onnx import select_onnx_providers; print('PyQt6 OK | ORT Available:', ort.get_available_providers(), '| Selected:', select_onnx_providers()); assert 'CPUExecutionProvider' in ort.get_available_providers()"
if errorlevel 1 (
    echo [ERROR] PyQt6 / ONNX Runtime verification failed inside %BUILD_VENV%.
    exit /b 1
)

echo.
echo ============================================================================
echo  [4/6] Compiling standalone protected PyQt6 Windows executable with Nuitka...
echo        Flags: --standalone --windows-console-mode=disable --enable-plugin=pyqt6
echo ============================================================================

if exist "dist\run.dist" rmdir /s /q "dist\run.dist"
if exist "dist\run.build" rmdir /s /q "dist\run.build"

"%VENV_PY%" -m nuitka ^
    --standalone ^
    --windows-console-mode=disable ^
    --enable-plugin=pyqt6 ^
    --assume-yes-for-downloads ^
    --show-progress ^
    --jobs=%NUITKA_JOBS% ^
    --lto=no ^
    --output-dir=dist ^
    --output-filename=ANPR_GateSystem.exe ^
    --product-name="ANPR Gate Access System" ^
    --file-version=1.0.0.0 ^
    --product-version=1.0.0.0 ^
    --include-package=app ^
    --include-package=app.gui ^
    --include-module=inference_onnx ^
    --include-module=license_check ^
    --include-module=onnxruntime.capi.onnxruntime_pybind11_state ^
    --user-package-configuration-file=nuitka_onnx_dml.yml ^
    --include-package=uvicorn.logging ^
    --include-package=uvicorn.loops ^
    --include-package=uvicorn.protocols ^
    --include-package=uvicorn.lifespan ^
    --include-package-data=onnxruntime ^
    --include-package-data=rapidocr_onnxruntime ^
    --include-package-data=fast_plate_ocr ^
    --include-data-file=models/license_plate_detector.onnx=models/license_plate_detector.onnx ^
    --include-data-file=models/vehicle_detector.onnx=models/vehicle_detector.onnx ^
    --include-data-file=LICENSE_NOTICE.txt=LICENSE_NOTICE.txt ^
    --include-data-file=.env.example=.env.example ^
    --nofollow-import-to=torch ^
    --nofollow-import-to=torchvision ^
    --nofollow-import-to=torchaudio ^
    --nofollow-import-to=ultralytics ^
    --nofollow-import-to=easyocr ^
    --nofollow-import-to=onnx ^
    --nofollow-import-to=sympy ^
    --nofollow-import-to=onnxruntime.tools ^
    --nofollow-import-to=onnxruntime.transformers ^
    --nofollow-import-to=onnxruntime.quantization ^
    --nofollow-import-to=onnxruntime.backend ^
    --nofollow-import-to=pygments.lexers ^
    --nofollow-import-to=sqlalchemy.dialects.postgresql ^
    --nofollow-import-to=sqlalchemy.dialects.mysql ^
    --nofollow-import-to=sqlalchemy.dialects.mssql ^
    --nofollow-import-to=sqlalchemy.dialects.oracle ^
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
echo  [5/6] Aligning VC++ 14.40+ runtime DLLs, verifying DirectML, writing .env...
echo ============================================================================

set "DIST_DIR=dist\run.dist"
if not exist "%DIST_DIR%\ANPR_GateSystem.exe" (
    echo [ERROR] Expected executable not found at %DIST_DIR%\ANPR_GateSystem.exe
    exit /b 1
)

REM Replace any older VS2019 msvcp140.dll (14.29) bundled by Nuitka with the
REM current VS2022+ (14.40+) VC++ runtime DLLs required by onnxruntime-directml.
for %%D in (msvcp140.dll msvcp140_1.dll vcruntime140.dll vcruntime140_1.dll) do (
    if exist "%SystemRoot%\System32\%%D" (
        copy /Y "%SystemRoot%\System32\%%D" "%DIST_DIR%\%%D" >nul
    )
)

REM Ensure DirectML.dll is present both in onnxruntime\capi\ and at bundle root
if not exist "%DIST_DIR%\DirectML.dll" (
    copy /Y "%ORT_CAPI_DIR%\DirectML.dll" "%DIST_DIR%\DirectML.dll" >nul
)
if not exist "%DIST_DIR%\onnxruntime\capi\DirectML.dll" (
    copy /Y "%ORT_CAPI_DIR%\DirectML.dll" "%DIST_DIR%\onnxruntime\capi\DirectML.dll" >nul
)

REM Verify DirectML.dll and core ONNX Runtime native binaries are present in output folder
for %%F in (
    "%DIST_DIR%\DirectML.dll"
    "%DIST_DIR%\onnxruntime\capi\DirectML.dll"
    "%DIST_DIR%\onnxruntime\capi\onnxruntime.dll"
    "%DIST_DIR%\onnxruntime\capi\onnxruntime_providers_shared.dll"
    "%DIST_DIR%\onnxruntime\capi\onnxruntime_pybind11_state.pyd"
) do (
    if not exist %%F (
        echo [ERROR] Required ONNX native binary missing from bundle: %%F
        exit /b 1
    )
    echo   Verified native binary: %%F
)

(
echo # Packaged Windows Runtime Configuration ^(ONNX Runtime DirectML/CPU + Native PyQt6 GUI^)
echo INFERENCE_BACKEND=onnx
echo DEVICE=auto
echo CAMERA_SOURCE=0
echo PLATE_ONNX_MODEL_PATH=models/license_plate_detector.onnx
echo VEHICLE_ONNX_MODEL_PATH=models/vehicle_detector.onnx
echo API_HOST=127.0.0.1
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
echo   Runtime Config    : %CD%\%DIST_DIR%\.env ^(update LICENSE_SERVER_URL ^& LICENSE_KEY^)
echo.
echo   NOTE: Original .pt weights and .py source files are NOT copied to dist\.
echo ============================================================================
exit /b 0
