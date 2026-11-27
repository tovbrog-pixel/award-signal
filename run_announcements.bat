@echo off
rem Award Signal - double-click to run Phase 1 on DoD daily announcements
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_announcements.ps1"
