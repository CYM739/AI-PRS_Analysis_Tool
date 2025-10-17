@echo off
SETLOCAL

REM Define the path to your portable Python executable
SET PYTHON_EXE=%~dp0python-embed\python.exe

ECHO ================================================================
ECHO  AI-PRS Analysis Tool - Setup
ECHO ================================================================
ECHO.

ECHO Step 1: Ensuring pip is installed...
REM Run the get-pip.py script to install pip if it's missing
"%PYTHON_EXE%" "%~dp0python-embed\get-pip.py"

ECHO.
ECHO Step 2: Installing required packages...
REM Now that pip is installed, use it to install the packages
"%PYTHON_EXE%" -m pip install --no-index --find-links="%~dp0packages" -r "%~dp0requirements.txt" --upgrade

ECHO.
ECHO Setup complete!
ECHO.

ECHO ================================================================
ECHO  Starting Application...
ECHO ================================================================

REM Run the Streamlit app using the portable python
"%PYTHON_EXE%" -m streamlit run "%~dp0/src/app.py"

ENDLOCAL
pause