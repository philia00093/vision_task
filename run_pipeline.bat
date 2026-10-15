@echo off
chcp 65001 > nul
REM Vision pipeline: green region detection + hand tracking
REM Training code lives under wersion_train\ (see wersion_train\README.md)
REM NOTE: keep REM comments ASCII; full-width CJK punctuation breaks cmd parsing.
REM   run_pipeline.bat --source test_1.png --save
REM   run_pipeline.bat --source 0
setlocal
set PY=D:\anaconda3\envs\yolo26\python.exe
if not exist "%PY%" (
  echo [错误] 找不到 yolo26 环境解释器 %PY%
  echo        请修改本文件里的 PY 变量
  exit /b 1
)
REM Inject package path via PYTHONPATH so the caller's working directory is unchanged
set "PYTHONPATH=%~dp0wersion_train;%PYTHONPATH%"
"%PY%" -m vision.run_pipeline %*
exit /b %ERRORLEVEL%
