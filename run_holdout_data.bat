@echo off
rem Award Signal - double-click to run the holdout award history (no returns)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_holdout_data.ps1"
