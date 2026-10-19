@echo off
chcp 65001 > nul
REM Open X-AnyLabeling for one class folder.
REM Usage (run from anywhere):
REM   annotate_parts.bat base
REM   annotate_parts.bat support
REM   annotate_parts.bat compass
REM   annotate_parts.bat top
REM NOTE: keep REM lines ASCII; full-width CJK punctuation breaks cmd parsing.
setlocal

set "CATEGORY=%~1"
if "%CATEGORY%"=="" set "CATEGORY=base"

set "PROJECT=D:\wang_robot\3_1\version_task\vision_task"
set "IMAGES=%PROJECT%\vision_yolo_train\datasets\annotate\parts\%CATEGORY%\images"
set "OUTPUT=%PROJECT%\vision_yolo_train\annotations"
set "LABELS=%PROJECT%\vision_yolo_train\configs\parts_labels.txt"

if not exist "%IMAGES%" (
  echo [错误] 找不到该类别目录：%IMAGES%
  echo        可选：base / support / compass / top
  goto :end
)
if not exist "%OUTPUT%" mkdir "%OUTPUT%"

echo 正在启动 X-AnyLabeling ...
echo   类别：%CATEGORY%
echo   图片：%IMAGES%
echo   标注：%OUTPUT%
echo.
echo 标注要点：每张图画一个矩形框；右侧标签选 %CATEGORY%
echo 快捷键：R 画框 / A 上一张 / D 下一张 / Ctrl+J 绘制-编辑模式切换 / Ctrl+S 保存
echo.

call conda activate anylabeling 2>nul
if errorlevel 1 (
  echo [提示] conda activate 不可用，改用 conda run
  conda run -n anylabeling xanylabeling --filename "%IMAGES%" --output "%OUTPUT%" --labels "%LABELS%"
) else (
  xanylabeling --filename "%IMAGES%" --output "%OUTPUT%" --labels "%LABELS%"
)

:end
call :maybe_pause
exit /b 0

:maybe_pause
REM Double-clicked windows keep the console open; a shell invocation does not pause.
echo %cmdcmdline% | find /i "%~nx0" >nul
if not errorlevel 1 pause
exit /b 0
