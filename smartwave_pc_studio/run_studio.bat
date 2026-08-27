@echo off
chcp 65001 > nul
title SmartWave Industrial Acoustic AI Studio
echo [SmartWave AI Studio] Starting Enterprise Acoustic AI Trainer...
python "%~dp0smartwave_studio_app.py"
pause
