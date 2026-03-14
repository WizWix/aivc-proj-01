@echo off
cd ..
call ".\.venv\Scripts\activate.bat"
echo ================================================================
py .\services\image_classification.py
pause
