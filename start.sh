#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# 启动 pdf2zh 在线翻译服务（内网 / 局域网）
# - 使用项目内虚拟环境 venv，不污染全局 Python
# - 前置：MTranServer 翻译后端需已运行，由 MTRANSERVER_ENDPOINT 指向
# ============================================================

# 设置终端窗口标题（xterm 兼容终端生效）
printf '\033]0;%s\007' 'pdf2zh 在线翻译服务'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install "$SCRIPT_DIR/."
    # 内网可改用镜像： pip install -i https://mirrors.aliyun.com/pypi/simple "$SCRIPT_DIR/."
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# --- 可选：内网离线资源（不联网下载模型/字体）---
# export PDF2ZH_DOCLAYOUT_MODEL=/path/to/doclayout_yolo_docstructbench_imgsz1024.onnx
# export NOTO_FONT_PATH=/path/to/GoNotoKurrent-Regular.ttf

# --- MTranServer 翻译后端地址（默认本机 8989）---
export MTRANSERVER_ENDPOINT="${MTRANSERVER_ENDPOINT:-http://127.0.0.1:8989}"
# export MTRANSERVER_API_TOKEN=your_token

# --- 仅暴露 mtranserver 翻译服务（默认即如此）---
export PDF2ZH_SERVICES="${PDF2ZH_SERVICES:-mtranserver}"

echo "Starting pdf2zh online service on 0.0.0.0:11008 ..."
exec pdf2zh --flask --host 0.0.0.0 --service mtranserver
