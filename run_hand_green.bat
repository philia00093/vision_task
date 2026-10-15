@echo off
chcp 65001 > nul
REM Hand detection + green region boxes (see wersion_train\README.md for training)
REM   run_hand_green.bat                                              live camera
REM   run_hand_green.bat --source wersion_train\images\parts\test_1.png --save
REM   run_hand_green.bat --green-lo 30 60 80 --min-area 0.01
REM NOTE: keep REM comments ASCII; full-width CJK punctuation breaks cmd parsing.
setlocal
set PY=D:\anaconda3\envs\yolo26\python.exe
if not exist "%PY%" (
  echo [错误] 找不到 yolo26 环境解释器 %PY%
  echo        请修改本文件里的 PY 变量
  exit /b 1
)
REM Inject package path via PYTHONPATH so the caller's working directory is unchanged
set "PYTHONPATH=%~dp0wersion_train;%PYTHONPATH%"
"%PY%" -m vision.hand_and_green %*
exit /b %ERRORLEVEL%
