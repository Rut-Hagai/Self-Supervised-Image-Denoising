@echo off
REM ===================================================================
REM  Trains all five methods, one after the other, under one identical
REM  protocol. Safe to stop and restart: every run resumes from its own
REM  last checkpoint and finished methods are skipped.
REM
REM    run_all.bat             20000 steps per method (the full run)
REM    run_all.bat 200         a quick check that everything works
REM ===================================================================
cd /d "%~dp0"

set STEPS=%1
if "%STEPS%"=="" set STEPS=20000

REM ---- find a working Python ----------------------------------------
REM  "python" may be the Microsoft Store placeholder, so each candidate is
REM  tested by running it; then the default install folders are searched.
set PYTHON=
call :try_cmd "py -3"
call :try_cmd "python"
call :try_cmd "python3"
if defined PYTHON goto :found

for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do call :try_cmd "%%~D\python.exe"
if defined PYTHON goto :found
for /d %%D in ("%ProgramFiles%\Python3*") do call :try_cmd "%%~D\python.exe"
if defined PYTHON goto :found
for /d %%D in ("C:\Python3*") do call :try_cmd "%%~D\python.exe"
if defined PYTHON goto :found

echo.
echo No working Python was found.
echo.
echo If Python is not installed: get it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer.
echo.
echo If it IS installed, re-run its installer, choose Modify, and tick
echo "Add Python to environment variables" on the Advanced Options page.
echo Then close this window and open a new one: a window that is already
echo open keeps the old PATH.
echo.
pause
exit /b 1

:try_cmd
if defined PYTHON exit /b 0
%~1 -c "import sys" >nul 2>nul
if errorlevel 1 exit /b 0
set PYTHON=%~1
exit /b 0

:found
echo Using: %PYTHON%
"%PYTHON%" --version 2>nul || %PYTHON% --version

echo.
echo === installing requirements ===
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 goto :pipfail

if not exist "data\Kodak" (
  echo.
  echo === fetching images and the authors' checkpoints ===
  %PYTHON% scripts\get_data.py
  if errorlevel 1 goto :datafail
)

if not exist results mkdir results

echo.
echo === training five methods, %STEPS% steps each ===
echo Started at %TIME%
for %%M in (supervised noise2noise noise2void neighbor2neighbor blind2unblind) do call :train %%M
echo.
echo === all done at %TIME% ===
echo Trained weights are in weights\ and the logs in results\.
pause
exit /b 0

:train
if exist "results\%~1_gauss25.done" (
  echo --- %~1 already finished, skipping
  exit /b 0
)
echo.
echo --- %~1   %TIME%
%PYTHON% -m src.train --method %~1 --steps %STEPS% --batch 4 --patch 64 ^
    --n-feature 16 --lr 1e-3 --test-dir data\Kodak ^
    --eval-every 2000 --ckpt-every 250 --resume --seed 0
if errorlevel 1 exit /b 0
type nul > "results\%~1_gauss25.done"
exit /b 0

:pipfail
echo.
echo Installing the Python packages failed. Scroll up for the reason.
pause
exit /b 1

:datafail
echo.
echo Downloading the data failed. It needs git and a network connection.
echo Get git from https://git-scm.com/download/win if it is missing.
pause
exit /b 1
