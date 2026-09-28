@echo off
setlocal
cd /d "%~dp0"
set NO_ALBUMENTATIONS_UPDATE=1
python -m pytest %*
exit /b %errorlevel%
