@echo off
cd ..
call ".\.venv\Scripts\activate.bat"
echo ================================================================
py .\services\ocr.py .\images\sample_korean_text.png
pause
