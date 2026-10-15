@echo off
REM 用 yolo26 专用环境下载 YOLO26 预训练权重到 ..\weights
REM 用法：download_weights.bat          只下 yolo26n.pt 和 yolo26.pt
REM       download_weights.bat --pose   额外下 yolo26n-pose.pt（手部关键点训练）
setlocal
set PY=D:\anaconda3\envs\yolo26\python.exe
if not exist "%PY%" (
  echo [错误] 找不到 yolo26 环境解释器：%PY%
  echo        请修改本文件里的 PY 变量。
  exit /b 1
)
"%PY%" "%~dp0download_weights.py" %*
exit /b %ERRORLEVEL%
