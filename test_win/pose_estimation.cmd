@echo off
cd ..
call ".\.venv\Scripts\activate.bat"
echo ================================================================
py .\services\pose_estimation.py --image .\images\default_pose_image.jpg
pause
