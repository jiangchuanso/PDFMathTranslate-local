import json, urllib.request, urllib.error

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def head(url, timeout=30):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "probe"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.headers.get("content-length", 0)), r.status
    except urllib.error.HTTPError as e:
        return 0, e.code
    except Exception as e:
        return 0, f"ERR {type(e).__name__}"

for name in ["jiangzhuo9357/opus-mt-en-zh-ct2", "jiangzhuo9357/opus-mt-zh-en-ct2"]:
    print("=" * 60)
    print(name)
    try:
        data = json.loads(get(f"https://huggingface.co/api/models/{name}"))
        sibs = [s.get("rfilename") for s in data.get("siblings", [])]
        print("  files:", sorted(sibs))
    except urllib.error.HTTPError as e:
        print("  HTTP", e.code, "(missing)")
        continue
    except Exception as e:
        print("  ERR", type(e).__name__, e)
        continue
    base = f"https://huggingface.co/{name}/resolve/main"
    for f in ["model.bin", "source.spm", "target.spm", "shared_vocabulary.json", "config.json", "vocab.json"]:
        size, st = head(f"{base}/{f}")
        print(f"  {f}: {size/1e6:.1f} MB  [{st}]")
    # show config.json content (may reveal quantization)
    try:
        cfg = get(f"{base}/config.json", timeout=20).decode("utf-8", "ignore")
        print("  config.json:", cfg[:400].replace("\n", " "))
    except Exception as e:
        print("  config.json unreadable:", e)

print("\n=== reference: float16 sizes for comparison ===")
for n in ["ooeoeo/opus-mt-en-zh-ct2-float16"]:
    s, st = head(f"https://huggingface.co/{n}/resolve/main/model.bin")
    print(f"  {n}/model.bin: {s/1e6:.1f} MB  [{st}]")
