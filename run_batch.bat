@echo off
rem Double-click to open the batch cell analysis dialog. Any arguments are
rem passed through, e.g.  run_batch.bat "D:\data\2026-08-17" --view
rem Uses the abf_analysis conda env directly, so no activation is needed.

set "PY=%USERPROFILE%\anaconda3\envs\abf_analysis\python.exe"
if not exist "%PY%" (
    echo Could not find the abf_analysis environment at:
    echo     %PY%
    echo Create it, or edit PY at the top of this file.
    pause
    exit /b 1
)

cd /d "%~dp0"
"%PY%" batch_cell_analysis.py %*
if errorlevel 1 (
    echo.
    echo The analysis stopped with an error; see the messages above.
    pause
)
