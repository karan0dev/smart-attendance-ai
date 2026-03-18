@echo off
TITLE CGC Jhanjeri - Smart Attendance System
color 0A

echo ===================================================
echo     INITIALIZING SMART ATTENDANCE SYSTEM
echo ===================================================
echo.
echo Checking for required libraries...
python -m pip install -r requirements.txt -q

echo.
echo Launching local server...
echo DO NOT CLOSE THIS WINDOW.
echo.

python -m streamlit run main.py

pause