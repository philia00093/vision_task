@echo off
chcp 65001 > nul
REM One-click pipeline: annotations -> YOLO labels -> train/val split -> check -> train
REM Run it (from anywhere):
REM   train_parts.bat                (default dataset: wersion_parts)
REM   train_parts.bat wersion_steps  (step recognition instead)
REM NOTE: keep REM lines ASCII; full-width CJK punctuation breaks cmd parsing.
setlocal

set "DATASET=%~1"
if "%DATASET%"=="" set "DATASET=wersion_parts"
set "PROJECT=D:\wang_robot\3_1\version_task\vision_task"
set "TOOLS=%PROJECT%\vision_yolo_train\tools"
set "ENTRY=%PROJECT%\vision_yolo_train\train_detect.py"

cd /d "%PROJECT%"
echo ============================================================
echo  数据集：%DATASET%
echo  工作目录：%PROJECT%
echo ============================================================
echo.

call conda activate yolo26 2>nul
if errorlevel 1 (
  echo [提示] conda activate 不可用，改用 conda run
  set "RUN=conda run -n yolo26 --no-capture-output python"
) else (
  set "RUN=python"
)

echo [1/4] 转换标注为 YOLO 标签 ...
%RUN% "%TOOLS%\convert_anylabeling.py" --data %DATASET%
if errorlevel 1 goto :failed

echo.
echo [2/4] 划分 train/val ...
%RUN% "%TOOLS%\split_dataset.py" --data %DATASET% --val-ratio 0.2
if errorlevel 1 goto :failed

echo.
echo [3/4] 数据集体检 ...
%RUN% "%TOOLS%\check_labels.py" --data %DATASET%
echo （体检不通过就先修数据，别急着训练）

echo.
echo [4/4] 开始训练 ...
%RUN% "%ENTRY%" --data %DATASET%
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo  全部完成。权重在 vision_yolo_train\runs\%DATASET%\weights\best.pt
echo ============================================================
goto :end

:failed
echo.
echo [失败] 上一步出错，已中止。请往上翻看报错信息。

:end
call :maybe_pause
exit /b 0

:maybe_pause
echo %cmdcmdline% | find /i "%~nx0" >nul
if not errorlevel 1 pause
exit /b 0
