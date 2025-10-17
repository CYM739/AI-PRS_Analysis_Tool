@echo off
SETLOCAL

REM Define the path to your portable Python executable
SET PYTHON_EXE=%~dp0python-embed\python.exe

ECHO ================================================================
ECHO  AI-PRS Analysis Tool (Education Edition) - Starting...
ECHO ================================================================
ECHO.

REM Run the Streamlit app for the Education Edition
"%PYTHON_EXE%" -m streamlit run "%~dp0/src/app_edu.py"

ENDLOCAL
pause