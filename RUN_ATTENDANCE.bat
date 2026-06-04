@echo off
TITLE CGC  - Smart Attendance System (Secure Boot)
color 0A

echo ===================================================
echo     INITIALIZING SECURE BIOMETRIC SYSTEM
echo ===================================================
echo.

:: STEP 1: Sandbox the Environment
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [INFO] First-time setup detected. Building secure isolated environment...
    python -m venv venv
)

:: STEP 2: Activate the Sandbox
echo [INFO] Engaging isolated virtual environment...
call venv\Scripts\activate.bat

:: STEP 3: Bypass C++ Hardware Limits (CRITICAL)
:: This prevents the fatal build crash on foreign laptops
echo [INFO] Injecting pre-compiled C++ biometric backend...
python -m pip install dlib-19.24.99-cp312-cp312-win_amd64.whl -q

:: STEP 4: Install Remaining Architecture
echo [INFO] Verifying deep learning architecture dependencies...
python -m pip install -r requirements.txt -q

echo.
echo [SUCCESS] System Boot Complete. Launching Control Panel...
echo DO NOT CLOSE THIS TERMINAL WINDOW.
echo.

:: STEP 5: Boot the Dash Board
python -m streamlit run main.py

pause