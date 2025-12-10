@echo off
setlocal

REM Set PYTHONPATH to include the current directory
set "PYTHONPATH=%CD%"

echo Running tests with coverage...
pytest

echo Generating HTML coverage report...
coverage html

echo Done. Report available at htmlcov/index.html