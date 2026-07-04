@echo off
REM Batch wrapper to run training from the repo virtualenv or system python.
REM Usage: train.bat --data-dir ..\datasets\ffpp --epochs 10 --batch-size 16 --lr 1e-4

SET REPO_ROOT=%~dp0..
SET REPO_ROOT=%REPO_ROOT:~0,-1%
cd /d "%REPO_ROOT%\ml"

if exist "%REPO_ROOT%\.venv\Scripts\activate.bat" (
    call "%REPO_ROOT%\.venv\Scripts\activate.bat"
) else (
    echo No virtualenv activation script found at %REPO_ROOT%\.venv\Scripts\activate.bat
    echo Using system Python from PATH.
)

python train_real.py %*

echo Training process finished.
pause