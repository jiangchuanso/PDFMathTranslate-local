import hashlib
import html
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from json import dumps, loads
from string import Template
from threading import Lock
from urllib.parse import unquote, urlparse
from uuid import uuid4

import requests
from flask import Flask, request, send_file

from pdf2zh import translate_stream
from pdf2zh.doclayout import ModelInstance

flask_app = Flask("pdf2zh")

# 内网部署唯一翻译后端（与 start.sh 的 PDF2ZH_SERVICES=mtranserver 保持一致）。
# 修改此处可切换默认后端（需对应后端已配置/已运行）。
DEFAULT_SERVICE = "mtranserver"

# URL 翻译结果缓存根目录（可用 PDF2ZH_CACHE_DIR 覆盖）。
_CACHE_ROOT = os.environ.get("PDF2ZH_CACHE_DIR") or os.path.join(
    tempfile.gettempdir(), "pdf2zh_url_cache"
)

# 内网单机部署：不依赖 celery/redis，用线程池 + 内存任务表实现异步翻译。
# 任务状态保持与原 celery 版兼容：PROGRESS / SUCCESS / FAILURE。
_TASKS: dict = {}
_LOCK = Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pdf2zh")


# ---------------------------------------------------------------------------
# URL 文件级缓存：以 URL 为键，磁盘路径按「scheme / host_port / 各级目录 / 文件名__短hash」
# 组织；meta.json 记录下载时的文件字节数（size）。再次请求时 HEAD 比对远端
# Content-Length，大小一致即直接复用已翻译的 mono/dual 结果，不再重译。
# ---------------------------------------------------------------------------
_UNSAFE_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def _url_cache_dir(url: str) -> str:
    """把 URL 映射为缓存目录：scheme/host:port/path/.../name__md5[:8]。"""
    p = urlparse(url)
    host = _UNSAFE_CHARS.sub("_", p.netloc).replace(":", "_")
    path = unquote(p.path).strip("/")
    parts = [seg for seg in path.split("/") if seg and seg not in (".", "..")]
    parts = [_UNSAFE_CHARS.sub("_", seg)[:120] for seg in parts]
    name = parts.pop() if parts else "index"
    key = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return os.path.join(_CACHE_ROOT, p.scheme or "http", host, *parts, f"{name}__{key}")


def _cache_lookup(url: str):
    """命中返回 (mono_bytes, dual_bytes)；未命中/大小变化返回 None。"""
    cache_dir = _url_cache_dir(url)
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = loads(f.read())
    except Exception:  # noqa: BLE001
        return None
    remote_size = None
    try:
        r = requests.head(url, timeout=10, allow_redirects=True)
        remote_size = r.headers.get("Content-Length")
    except Exception:  # noqa: BLE001
        remote_size = None
    # HEAD 成功且有大小：不一致 → 文件已更新，需重译；HEAD 失败则保守视为命中
    if remote_size is not None and str(meta.get("size")) != remote_size:
        return None
    try:
        with open(os.path.join(cache_dir, "mono.pdf"), "rb") as f:
            mono = f.read()
        with open(os.path.join(cache_dir, "dual.pdf"), "rb") as f:
            dual = f.read()
    except Exception:  # noqa: BLE001
        return None
    print(f"[cache] hit: {url} (size={meta.get('size')})")
    return mono, dual


def _download_url(url: str) -> tuple[bytes, int]:
    r = requests.get(url, timeout=300, allow_redirects=True)
    r.raise_for_status()
    content = r.content
    if not content:
        raise ValueError("empty response body")
    return content, len(content)


def _save_cache(cache_dir: str, url: str, size: int, doc_mono: bytes, doc_dual: bytes):
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "mono.pdf"), "wb") as f:
            f.write(doc_mono)
        with open(os.path.join(cache_dir, "dual.pdf"), "wb") as f:
            f.write(doc_dual)
        meta = {
            "url": url,
            "size": size,
            "translated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
            f.write(dumps(meta, ensure_ascii=False))
        print(f"[cache] saved: {cache_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"[cache] save failed: {e}")


def _run_translate(
    task_id: str,
    stream: bytes,
    args: dict,
    cache_dir: str = None,
    url: str = None,
    file_size: int = 0,
):
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
        if cache_dir:
            _save_cache(cache_dir, url or "", file_size, doc_mono, doc_dual)
    except Exception as e:  # noqa: BLE001
        entry["state"] = "FAILURE"
        entry["error"] = str(e)


@flask_app.route("/v1/translate", methods=["POST"])
def create_translate_tasks():
    args = loads(request.form.get("data") or "{}")
    # URL 优先：form 字段 url 或 data JSON 里的 url（where 等外部应用按此调用）
    url = (request.form.get("url") or args.pop("url", "") or "").strip()
    args.setdefault("service", DEFAULT_SERVICE)
    args.setdefault("lang_out", "zh")
    force = bool(args.pop("force", False))

    task_id = uuid4().hex

    if url and not force:
        hit = _cache_lookup(url)
        if hit:
            with _LOCK:
                _TASKS[task_id] = {
                    "state": "SUCCESS",
                    "info": {"n": 0, "total": 0},
                    "cached": True,
                    "docs": hit,
                }
            return {"id": task_id, "cached": True}

    if url:
        try:
            stream, file_size = _download_url(url)
        except Exception as e:  # noqa: BLE001
            return {"error": f"download failed: {e}"}, 400
        cache_dir = _url_cache_dir(url)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "source.pdf"), "wb") as f:
                f.write(stream)
        except Exception as e:  # noqa: BLE001
            print(f"[cache] persist source failed: {e}")
            cache_dir = None
    else:
        file = request.files.get("file")
        if file is None:
            return {"error": "no file or url provided"}, 400
        stream = file.stream.read()
        file_size = len(stream)
        cache_dir = None

    with _LOCK:
        _TASKS[task_id] = {"state": "PROGRESS", "info": {"n": 0, "total": 0}}
    _executor.submit(_run_translate, task_id, stream, args, cache_dir, url, file_size)
    return {"id": task_id, "cached": False}


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
    <label>方式一：文件 URL（跨服务器场景，服务端下载；同 URL 且文件大小未变将直接复用缓存结果）</label>
    <input type="text" id="url" placeholder="http://192.168.0.17:8080/docs/xxx.pdf">
    <label style="margin-top:16px">方式二：本地上传 PDF 文件</label>
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
  const url = $("url").value.trim();
  const f = $("file").files[0];
  if(!url && !f){ setErr("请填写文件 URL 或选择 PDF 文件"); return; }
  const fd = new FormData();
  if(url){
    fd.append("url", url);
  } else {
    fd.append("file", f);
  }
  const data = {
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
    if(!r.ok){
      const j = await r.json().catch(()=>({}));
      setErr("提交失败: "+r.status+" "+(j.error||""));
      $("go").disabled=false; return;
    }
    const {id, cached} = await r.json();
    setOk(cached ? "缓存命中（文件未变化），直接取结果…" : "已提交，排队中…");
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
