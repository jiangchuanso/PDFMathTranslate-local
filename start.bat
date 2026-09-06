@echo off
setlocal
REM ============================================================
REM 启动 pdf2zh 在线翻译服务（内网 / 局域网）
REM 前置条件：MTranServer 翻译后端需已运行，并通过下面的
REM           MTRANSERVER_ENDPOINT 指向其地址。
REM ============================================================

REM --- 可选：内网离线资源（不联网下载模型/字体）---
REM set PDF2ZH_DOCLAYOUT_MODEL=C:\path\to\doclayout_yolo_docstructbench_imgsz1024.onnx
REM set NOTO_FONT_PATH=C:\path\to\GoNotoKurrent-Regular.ttf

REM --- MTranServer 翻译后端地址（默认本机 8989）---
set MTRANSERVER_ENDPOINT=http://127.0.0.1:8989
REM set MTRANSERVER_API_TOKEN=your_token

REM --- 仅暴露 mtranserver 翻译服务（默认即如此）---
set PDF2ZH_SERVICES=mtranserver

echo Starting pdf2zh online service on 0.0.0.0:11008 ...
pdf2zh --flask --host 0.0.0.0 --service mtranserver
endlocal
