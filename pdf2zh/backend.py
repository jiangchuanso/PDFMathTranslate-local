from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from json import loads
from string import Template
from threading import Lock
from uuid import uuid4

from flask import Flask, request, send_file

from pdf2zh import translate_stream
from pdf2zh.doclayout import ModelInstance

flask_app = Flask("pdf2zh")

# 内网单机部署：不依赖 celery/redis，用线程池 + 内存任务表实现异步翻译。
# 任务状态保持与原 celery 版兼容：PROGRESS / SUCCESS / FAILURE。
_TASKS: dict = {}
_LOCK = Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pdf2zh")


def _run_translate(task_id: str, stream: bytes, args: dict):
    entry = _TASKS[task_id]

    def progress_bar(t):
        with _LOCK:
            entry["info"] = {"n": t.n, "total": t.total}
        print(f"Translating {t.n} / {t.total} pages")

    try:
        if "prompt" in args:
            args["prompt"] = Template(args["prompt"])
        doc_mono, doc_dual = translate_stream(
            stream,
            callback=progress_bar,
            model=ModelInstance.value,
            **args,
        )
        entry["docs"] = (doc_mono, doc_dual)
        entry["state"] = "SUCCESS"
    except Exception as e:  # noqa: BLE001
        entry["state"] = "FAILURE"
        entry["error"] = str(e)


@flask_app.route("/v1/translate", methods=["POST"])
def create_translate_tasks():
    file = request.files["file"]
    stream = file.stream.read()
    args = loads(request.form.get("data") or "{}")
    task_id = uuid4().hex
    with _LOCK:
        _TASKS[task_id] = {"state": "PROGRESS", "info": {"n": 0, "total": 0}}
    _executor.submit(_run_translate, task_id, stream, args)
    return {"id": task_id}


@flask_app.route("/v1/translate/<task_id>", methods=["GET"])
def get_translate_task(task_id: str):
    entry = _TASKS.get(task_id)
    if entry is None:
        return {"error": "task not found"}, 404
    resp = {"state": entry["state"]}
    if entry["state"] == "PROGRESS":
        resp["info"] = entry["info"]
    elif entry["state"] == "FAILURE":
        resp["error"] = entry["error"]
    return resp


@flask_app.route("/v1/translate/<task_id>", methods=["DELETE"])
def delete_translate_task(task_id: str):
    with _LOCK:
        _TASKS.pop(task_id, None)
    return {"state": "removed"}


@flask_app.route("/v1/translate/<task_id>/<format>")
def get_translate_result(task_id: str, format: str):
    entry = _TASKS.get(task_id)
    if entry is None:
        return {"error": "task not found"}, 404
    if entry["state"] == "PROGRESS":
        return {"error": "task not finished"}, 400
    if entry["state"] != "SUCCESS":
        return {"error": entry.get("error") or "task failed"}, 400
    doc_mono, doc_dual = entry["docs"]
    to_send = doc_mono if format == "mono" else doc_dual
    return send_file(BytesIO(to_send), "application/pdf")


if __name__ == "__main__":
    flask_app.run()
