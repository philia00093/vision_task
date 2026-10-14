@echo off
chcp 65001 > nul
REM 建立 wersion_train 数据集骨架（images/labels 的 train/val 目录 + 类别表 + 数据集 yaml 模板）
REM 幂等：已存在的文件不会覆盖
setlocal
set PY=D:\anaconda3\envs\yolo26\python.exe
if not exist "%PY%" (
  echo [错误] 找不到 yolo26 环境解释器：%PY%
  echo        请修改本文件里的 PY 变量，或直接用 python 运行同名 .py
  exit /b 1
)
"%PY%" "%~dp0init_datasets.py" %*
exit /b %ERRORLEVEL%
