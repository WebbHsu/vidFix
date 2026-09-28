@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  echo Using .venv:
  ".venv\Scripts\python.exe" -c "import sys,torch; print(sys.executable); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  ".venv\Scripts\python.exe" app.py
) else (
  echo ERROR: .venv not found. Use Python 3.12 venv, not system Python 3.14.
  pause
  exit /b 1
)
pause
