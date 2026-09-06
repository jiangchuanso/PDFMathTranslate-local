@echo off
setlocal
title pdf2zh 在线翻译服务
REM ============================================================
REM 启动 pdf2zh 在线翻译服务（内网 / 局域网）
REM - 使用项目内虚拟环境 venv，不污染全局 Python
REM - 前置：MTranServer 翻译后端需已运行，由 MTRANSERVER_ENDPOINT 指向
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"

REM 虚拟环境不存在则创建并安装本仓库
if not exist "%VENV_DIR%\Scripts\python.exe" (
    python -m venv "%VENV_DIR%"
    call "%VENV_DIR%\Scripts\activate.bat"
    pip install --upgrade pip
    pip install "%SCRIPT_DIR%."
    REM 内网可改用镜像： pip install -i https://mirrors.aliyun.com/pypi/simple "%SCRIPT_DIR%."
) else (
    call "%VENV_DIR%\Scripts\activate.bat"
)

REM --- 可选：内网离线资源（不联网下载模型/字体）---
REM set PDF2ZH_DOCLAYOUT_MODEL=C:\path\to\doclayout_yolo_docstructbench_imgsz1024.onnx
REM set NOTO_FONT_PATH=C:\path\to\GoNotoKurrent-Regular.ttf

REM --- MTranServer 翻译后端地址（默认本机 8989）---
set "MTRANSERVER_ENDPOINT=http://127.0.0.1:8989"
REM set MTRANSERVER_API_TOKEN=your_token

REM --- 仅暴露 mtranserver 翻译服务（默认即如此）---
set "PDF2ZH_SERVICES=mtranserver"

echo Starting pdf2zh online service on 0.0.0.0:11008 ...
pdf2zh --flask --host 0.0.0.0 --service mtranserver
endlocal
