@echo off
chcp 65001 > nul
REM 只做两件事：识别手 + 框绿色区域
REM   run_hand_green.bat                                       实时相机
REM   run_hand_green.bat --source 图片.png --save
REM   run_hand_green.bat --green-lo 30 60 80 --min-area 0.01
setlocal
set PY=D:\anaconda3\envs\yolo26\python.exe
if not exist "%PY%" (
  echo [错误] 找不到 yolo26 环境解释器：%PY%
  echo        请修改本文件里的 PY 变量。
  exit /b 1
)
"%PY%" "%~dp0src\hand_and_green.py" %*
exit /b %ERRORLEVEL%
