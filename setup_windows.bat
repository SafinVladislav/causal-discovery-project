@echo off
title Causal Discovery Project - Setup

echo.
echo ==================================================
echo    Causal Discovery Project - Setup
echo ==================================================
echo.

:: Check if python exists and is in PATH
python --version >nul 2>&1
if %errorlevel% neq 0 (
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Python is not installed or not in PATH!
        echo.
        echo Please install Python from:
        echo     https://www.python.org/downloads/
        echo.
        echo Important: During installation, check the box
        echo            "Add Python to PATH"
        echo.
        pause
        exit /b 1
    ) else (
        set PYTHON=python3
    )
) else (
    set PYTHON=python
)

for /f "tokens=2" %%v in ('%PYTHON% --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%

if exist venv (
    echo Virtual environment folder exists - checking if it's healthy...
    venv\Scripts\python -m pip --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo   Existing venv looks healthy - reusing it.
    ) else (
        echo   Existing venv is broken or incomplete - deleting and recreating...
        rd /s /q venv
    )
)

if not exist venv (
    echo.
    echo Creating virtual environment...
    %PYTHON% -m venv venv || goto :error
) else (
    echo.
    echo Virtual environment already exists - reusing it.
)

call venv\Scripts\activate || goto :error

echo.
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1 || rem

echo.
echo Installing project dependencies...
pip install -r requirements.txt --upgrade || goto :error

call deactivate || rem

echo.
echo ==================================================
echo    SUCCESS! Everything is ready
echo ==================================================
echo.
echo To run the simulation later, just:
echo    Open command prompt in this folder and run:
echo       venv\Scripts\activate
echo       python run_simulation.py
echo       deactivate
echo.
pause
exit /b 0

:error
echo.
echo [ERROR] Something went wrong during setup.
echo Check the error messages above.
pause
exit /b 1