@echo off
cd ..
call ".\.venv\Scripts\activate.bat"
echo ================================================================
py .\services\face_recognition.py --img1 .\images\tom_cruise_1.jpg --img2 .\images\tom_cruise_2.jpg
echo ================================================================
py .\services\face_recognition.py --img1 .\images\tom_cruise_1.jpg --img2 .\images\brad_pitt.jpg
pause
