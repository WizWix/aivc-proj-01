@echo off
cd "%~dp0"
call ".venv\Scripts\activate.bat"
echo ================================================================
py main.py
pause
