import html
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from json import dumps, loads
from string import Template
from threading import Lock
from uuid import uuid4

from flask import Flask, request, send_file

from pdf2zh import translate_stream
from pdf2zh.doclayout import ModelInstance

flask_app = Flask("pdf2zh")

# 内网部署唯一翻译后端（与 start.sh 的 PDF2ZH_SERVICES=mtranserver 保持一致）。
# 修改此处可切换默认后端（需对应后端已配置/已运行）。
DEFAULT_SERVICE = "mtranserver"

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


_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDFMathTranslate 翻译</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --accent:#38bdf8; --text:#e2e8f0; --muted:#94a3b8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, "Microsoft YaHei", sans-serif; background:var(--bg); color:var(--text); }
  .wrap { max-width: 720px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .card { background: var(--card); border-radius: 12px; padding: 20px; }
  label { display:block; font-size: 13px; color: var(--muted); margin: 12px 0 4px; }
  input[type=file], input[type=text], input[type=number] {
    width:100%; padding: 9px 11px; border-radius: 8px; border:1px solid #334155;
    background:#0b1220; color:var(--text); font-size:14px;
  }
  .row { display:flex; gap:12px; }
  .row > div { flex:1; }
  button {
    margin-top:18px; width:100%; padding:11px; border:0; border-radius:8px;
    background:var(--accent); color:#04212e; font-weight:600; font-size:15px; cursor:pointer;
  }
  button:disabled { opacity:.5; cursor:not-allowed; }
  #status { margin-top:16px; font-size:14px; min-height:22px; }
  .bar { height:8px; background:#0b1220; border-radius:6px; overflow:hidden; margin-top:8px; }
  .bar > i { display:block; height:100%; width:0; background:var(--accent); transition:width .3s; }
  .dl { display:flex; gap:12px; margin-top:16px; }
  .dl a {
    flex:1; text-align:center; padding:10px; border-radius:8px; background:#0b1220;
    color:var(--accent); text-decoration:none; font-size:14px; border:1px solid #334155;
  }
  .err { color:#f87171; }
  .ok { color:#4ade80; }
</style>
</head>
<body>
<div class="wrap">
  <h1>PDF 翻译服务</h1>
  <div class="sub">翻译后端：__SERVICE__（MTranServer）</div>
  <div class="card">
    <label>PDF 文件</label>
    <input type="file" id="file" accept="application/pdf">
    <div class="row">
      <div>
        <label>源语言 (lang_in，留空自动)</label>
        <input type="text" id="lang_in" placeholder="如 zh / en / ja">
      </div>
      <div>
        <label>目标语言 (lang_out)</label>
        <input type="text" id="lang_out" placeholder="如 zh / en" value="zh">
      </div>
    </div>
    <div class="row">
      <div>
        <label>线程数 (thread)</label>
        <input type="number" id="thread" value="4" min="1" max="16">
      </div>
      <div>
        <label>跳过字体子集化</label>
        <input type="text" id="skip" placeholder="true / 留空" >
      </div>
    </div>
    <button id="go">开始翻译</button>
    <div id="status"></div>
    <div class="bar" id="barwrap" style="display:none"><i id="bar"></i></div>
    <div class="dl" id="dl" style="display:none">
      <a id="dl_mono" download>下载 单语版</a>
      <a id="dl_dual" download>下载 双语版</a>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const status = $("status");
function setErr(m){ status.className="err"; status.textContent=m; }
function setOk(m){ status.className="ok"; status.textContent=m; }

$("go").onclick = async () => {
  const f = $("file").files[0];
  if(!f){ setErr("请先选择 PDF 文件"); return; }
  const fd = new FormData();
  fd.append("file", f);
  const data = {
    service: "__SERVICE__",
    lang_in: $("lang_in").value.trim(),
    lang_out: $("lang_out").value.trim(),
    thread: parseInt($("thread").value)||4,
  };
  const skip = $("skip").value.trim().toLowerCase();
  if(skip === "true") data.skip_subset_fonts = true;
  fd.append("data", JSON.stringify(data));

  $("go").disabled = true; $("dl").style.display="none";
  $("barwrap").style.display="block"; $("bar").style.width="0%";
  try {
    const r = await fetch("/v1/translate", {method:"POST", body:fd});
    if(!r.ok){ setErr("提交失败: "+r.status); $("go").disabled=false; return; }
    const {id} = await r.json();
    setOk("已提交，排队中…");
    poll(id);
  } catch(e){ setErr("网络错误: "+e); $("go").disabled=false; }
};

async function poll(id){
  try {
    const r = await fetch("/v1/translate/"+id);
    const j = await r.json();
    if(j.state === "PROGRESS"){
      const {n=0,total=0} = j.info||{};
      const pct = total? Math.round(n/total*100):0;
      $("bar").style.width = pct+"%";
      status.className=""; status.textContent = `翻译中 ${n} / ${total} 页`;
      setTimeout(()=>poll(id), 1500);
    } else if(j.state === "SUCCESS"){
      $("bar").style.width="100%"; setOk("翻译完成");
      $("dl").style.display="flex";
      $("dl_mono").href = `/v1/translate/${id}/mono`;
      $("dl_dual").href = `/v1/translate/${id}/dual`;
      $("go").disabled=false;
    } else if(j.state === "FAILURE"){
      setErr("翻译失败: "+(j.error||"未知错误")); $("go").disabled=false;
    } else {
      setErr("异常状态: "+j.state); $("go").disabled=false;
    }
  } catch(e){ setErr("轮询错误: "+e); $("go").disabled=false; }
}
</script>
</body>
</html>
"""


@flask_app.route("/", methods=["GET"])
def index():
    return _INDEX_HTML.replace("__SERVICE__", html.escape(DEFAULT_SERVICE))


if __name__ == "__main__":
    flask_app.run()
