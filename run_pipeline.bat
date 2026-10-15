@echo off
REM 用 yolo26 环境跑视觉流水线（区域检测 + 手部跟踪）
REM 所有参数原样传给 src\run_pipeline.py，例如：
REM   run_pipeline.bat --source "test_1.png" --save
REM   run_pipeline.bat --hand weights\yolo26n-pose.pt --save --save-baseline
setlocal
set PY=D:\anaconda3\envs\yolo26\python.exe
if not exist "%PY%" (
  echo [错误] 找不到 yolo26 环境解释器：%PY%
  echo        请修改本文件里的 PY 变量。
  exit /b 1
)
"%PY%" "%~dp0src\run_pipeline.py" %*
exit /b %ERRORLEVEL%
