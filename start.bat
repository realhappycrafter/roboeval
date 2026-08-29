@echo off
chcp 65001 >nul
title RoboEval Server
cd /d "%~dp0backend"
echo ============================================
echo   RoboEval - 具身智能策略评测平台
echo   启动后浏览器访问 http://127.0.0.1:8000
echo ============================================
"C:\Users\nel92\.workbuddy\binaries\python\envs\default\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
