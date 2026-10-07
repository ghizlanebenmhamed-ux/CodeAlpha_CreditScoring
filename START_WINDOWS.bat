@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (set "PYTHON_CMD=python") else (set "PYTHON_CMD=py")
if exist .venv\Scripts\python.exe goto install
%PYTHON_CMD% -m venv .venv
if errorlevel 1 goto failed
:install
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe app.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Startup failed. Read the error above. Python with Tkinter is required.
pause
