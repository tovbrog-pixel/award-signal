@echo off
rem Award Signal - double-click to run the version 3 holdout test
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_holdout.ps1"
