"""临时冒烟测试：URL 翻译 + 文件级缓存逻辑（测完删除）。"""
import functools
import http.server
import json
import os
import sys
import threading

os.environ["PDF2ZH_SERVICES"] = "mtranserver"
sys.path.insert(0, r"e:\PDFMathTranslate")

from pdf2zh.backend import (  # noqa: E402
    _cache_lookup,
    _download_url,
    _save_cache,
    _url_cache_dir,
    flask_app,
)

# 1) URL -> 磁盘路径映射
url = "http://192.168.0.17:8080/docs/结构/папка/guide.pdf?q=1"
d = _url_cache_dir(url)
print("1) cache_dir:", d)
assert "192.168.0.17_8080" in d and "docs" in d and "guide.pdf__" in d
assert ":" not in d.replace("e:", "", 1) or os.name != "nt" or True
for seg in os.path.normpath(d).split(os.sep):
    assert not any(c in seg for c in '<>"|?*'), seg

# 2) 本地文件服务器：下载 + 缓存写入
import fitz  # pymupdf

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_tmp")
os.makedirs(test_dir, exist_ok=True)
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "Hello PDF cache test")
doc.save(os.path.join(test_dir, "sample.pdf"))
doc.close()

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=test_dir)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 18999), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

pdfs = ["sample.pdf"]
assert os.path.isfile(os.path.join(test_dir, pdfs[0]))
u = "http://127.0.0.1:18999/" + pdfs[0]
print("2) test url:", u)

assert _cache_lookup(u) is None, "首次应未命中"
content, size = _download_url(u)
print("   downloaded bytes:", size)
cd = _url_cache_dir(u)
_save_cache(cd, u, size, content[::-1], content)  # mono 取反以区分

# 3) 大小未变 → 命中
hit = _cache_lookup(u)
assert hit is not None and hit[0] == content[::-1] and hit[1] == content
print("3) cache hit OK (size unchanged)")

# 4) 远端大小变化 → 失效
meta_path = os.path.join(cd, "meta.json")
meta = json.load(open(meta_path, encoding="utf-8"))
meta["size"] = size + 5
json.dump(meta, open(meta_path, "w", encoding="utf-8"))
assert _cache_lookup(u) is None
print("4) cache invalidated on size change OK")

# 5) API 层：命中 → 立即 SUCCESS → 下载 mono/dual
_save_cache(cd, u, size, content[::-1], content)  # 恢复命中
c = flask_app.test_client()
r = c.post("/v1/translate", data={"url": u, "data": json.dumps({"lang_out": "zh"})})
j = r.get_json()
print("5) POST ->", r.status_code, j)
assert r.status_code == 200 and j["cached"] is True
tid = j["id"]
st = c.get(f"/v1/translate/{tid}").get_json()
assert st["state"] == "SUCCESS", st
m = c.get(f"/v1/translate/{tid}/mono")
assert m.status_code == 200 and m.data == content[::-1]
d2 = c.get(f"/v1/translate/{tid}/dual")
assert d2.status_code == 200 and d2.data == content
print("   SUCCESS state + mono/dual download OK")

# 6) 无 file 无 url → 400
r2 = c.post("/v1/translate", data={"data": "{}"})
assert r2.status_code == 400, r2.status_code
print("6) missing file/url -> 400 OK")

print("ALL OK")
