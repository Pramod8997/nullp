@echo off
setlocal
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0scripts\make_runner.py" %*
) else (
    python "%~dp0scripts\make_runner.py" %*
)
endlocal
