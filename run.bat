@echo off
REM AI Smart Four-Wheel Drive Car Project Runner for Windows
REM This script sets up and runs the project

echo Starting AI Smart Four-Wheel Drive Car...

REM Check if Python 3 is installed
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed. Please install Python and try again.
    exit /b 1
)

REM Check if pip is installed
pip --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo pip is not installed. Please install pip and try again.
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create required directories
echo Creating required directories...
if not exist static\css mkdir static\css
if not exist static\js mkdir static\js
if not exist templates mkdir templates
if not exist slam mkdir slam

REM Check if simulation mode is requested
set SIMULATION_FLAG=
if "%1"=="--simulation" (
    set SIMULATION_FLAG=--simulation
    echo Running in simulation mode (without hardware)
) else (
    echo Running in normal mode (with hardware if available)
)

REM Run the application
echo Starting the application...
python app.py %SIMULATION_FLAG%

REM Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat 