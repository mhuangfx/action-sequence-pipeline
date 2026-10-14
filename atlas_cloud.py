#!/usr/bin/env python3
"""Atlas Cloud API client — unified image/video generation (backup generator).

Atlas Cloud (https://www.atlascloud.ai) is an OpenAI-style aggregator exposing
300+ image/video models behind one key. We use it as the API-based BACKUP video
+ stills generator once the CCP browser-automation credit pools (Hailuo /
Dreamina) are exhausted.

Async REST contract (verified live 2026-06-25):
  POST {BASE}/model/generateImage   body: Input  -> {"data":{"id":...}} OR {"id":...}
  POST {BASE}/model/generateVideo   body: Input  -> {"data":{"id":...}}
  GET  {BASE}/model/prediction/{id}              -> {status, outputs:[url...]}
  GET  {BASE}/models                             -> full model catalog

Auth: `Authorization: Bearer <ATLAS_CLOUD_API>` from .env.
NOTE: the edge is behind Cloudflare which 403s (error 1010) on urllib's TLS
fingerprint — a browser User-Agent is required, so we shell out to curl.

Reference images may be passed as https URLs, `data:` base64 URIs, or
`asset://<id>`. We inline local files as base64 data URIs (refs are ~1.5MB,
well under the 30MB/ref limit).
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import time

BASE = "https://api.atlascloud.ai/api/v1"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_key():
    env = os.path.join(HERE, ".env")
    txt = open(env).read() if os.path.exists(env) else ""
    m = re.search(r"ATLAS_CLOUD_API=(\S+)", txt)
    key = (m.group(1) if m else os.environ.get("ATLAS_CLOUD_API", "")).strip()
    if not key:
        sys.exit("ATLAS_CLOUD_API not found in .env or environment")
    return key


def _curl(method, url, key, body=None, binary_out=None):
    """Shell out to curl (Cloudflare blocks urllib's JA3). Returns parsed JSON,
    or writes bytes to binary_out and returns the path."""
    cmd = ["curl", "-sS", "--fail-with-body", "-X", method, url,
           "-H", f"Authorization: Bearer {key}", "-H", f"User-Agent: {UA}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "--data-binary", "@-"]
    if binary_out:
        cmd += ["-o", binary_out]
    inp = json.dumps(body).encode() if body is not None else None
    r = subprocess.run(cmd, input=inp, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl {method} {url} failed rc={r.returncode}: "
                           f"{r.stderr.decode()[:300]} {r.stdout.decode()[:500]}")
    if binary_out:
        return binary_out
    out = r.stdout.decode()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"non-JSON response from {url}: {out[:500]}")


def file_to_datauri(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:{mime};base64,{b64}"


def as_ref(ref):
    """Accept a local path, http(s) URL, data: URI, or asset:// ref."""
    if ref.startswith(("http://", "https://", "data:", "asset://")):
        return ref
    if not os.path.exists(ref):
        sys.exit(f"reference not found: {ref}")
    return file_to_datauri(ref)


def upload_media(path, key=None):
    """Upload a local file (POST /model/uploadMedia, multipart 'file') -> a TEMPORARY https url.
    Needed for reference_videos: unlike reference_images, Seedance rejects base64 data URIs for
    videos ("reference_video must be provided as a web url"). The returned url is temporary —
    use it within the same job. Docs: https://www.atlascloud.ai/docs/en/upload-files"""
    key = key or load_key()
    r = subprocess.run(
        ["curl", "-sS", "--fail-with-body", "-X", "POST", f"{BASE}/model/uploadMedia",
         "-H", f"Authorization: Bearer {key}", "-H", f"User-Agent: {UA}",
         "-F", f"file=@{path}"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"uploadMedia {path} failed rc={r.returncode}: "
                           f"{r.stderr.decode()[:300]} {r.stdout.decode()[:500]}")
    out = r.stdout.decode()
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"uploadMedia non-JSON response: {out[:300]}")
    data = d.get("data", d) or {}
    url = d.get("url") or d.get("download_url") or data.get("url") or data.get("download_url")
    if not url:
        raise RuntimeError(f"uploadMedia: no url in response: {json.dumps(d)[:300]}")
    return url


def as_video_ref(ref, key=None):
    """Like as_ref but for reference_videos, which require a web url or asset:// (NOT base64):
    a local file is uploaded via uploadMedia to obtain a temporary url."""
    if ref.startswith(("http://", "https://", "asset://")):
        return ref
    if ref.startswith("data:"):
        raise RuntimeError("reference_videos cannot be a data: URI — pass a local path, https url, or asset://")
    if not os.path.exists(ref):
        sys.exit(f"reference video not found: {ref}")
    return upload_media(ref, key=key)


# ------------------------------------------------------------- daily spend cap
# Atlas bills real dollars per asset and (since it became the primary generator,
# late June) runs unattended nightly — without a cap, a runaway retry loop is an
# uncapped cost. Every submit records an ESTIMATED cost in a local ledger and
# refuses (loudly) once today's estimate reaches the cap. Estimates are
# intentionally conservative (rounded up); the cap is an emergency brake, not
# accounting. Override per-run with ATLAS_SPEND_CAP_USD=0 to disable (not
# recommended) or raise the config value.

SPEND_LEDGER = os.path.join(HERE, "output", "atlas_spend.jsonl")
GENERATORS_CONFIG = os.path.join(HERE, "config", "generators.json")
DEFAULT_DAILY_CAP_USD = 20.0
_VIDEO_USD_PER_SEC = 0.112          # seedance-2.0 @720p
_VIDEO_USD_PER_SEC_LITE = 0.056     # -fast / -mini variants
_IMAGE_USD = {"gpt-image-2": 0.01, "nano-banana": 0.14}
_IMAGE_USD_DEFAULT = 0.14


def spend_cap_usd():
    env_cap = os.environ.get("ATLAS_SPEND_CAP_USD")
    if env_cap:
        try:
            return float(env_cap)
        except ValueError:
            pass
    try:
        cfg = json.load(open(GENERATORS_CONFIG))
        cap = (cfg.get("generators", {}).get("atlas", {}) or {}).get("daily_spend_cap_usd")
        if isinstance(cap, (int, float)):
            return float(cap)
    except Exception:
        pass
    return DEFAULT_DAILY_CAP_USD


def estimate_cost(endpoint, payload):
    model = (payload.get("model") or "").lower()
    if "generatevideo" in endpoint.lower():
        dur = payload.get("duration") or 5
        rate = (_VIDEO_USD_PER_SEC_LITE if ("fast" in model or "mini" in model)
                else _VIDEO_USD_PER_SEC)
        return round(dur * rate, 3)
    for frag, usd in _IMAGE_USD.items():
        if frag in model:
            return usd
    return _IMAGE_USD_DEFAULT


def spent_today(now=None):
    today = time.strftime("%Y-%m-%d", time.gmtime(now or time.time()))
    total = 0.0
    try:
        with open(SPEND_LEDGER, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if (rec.get("ts") or "").startswith(today):
                    total += rec.get("est_usd") or 0.0
    except FileNotFoundError:
        pass
    return round(total, 3)


def _record_spend(endpoint, payload, est):
    os.makedirs(os.path.dirname(SPEND_LEDGER), exist_ok=True)
    with open(SPEND_LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": endpoint, "model": payload.get("model"),
            "est_usd": est}) + "\n")


def check_spend_cap(endpoint, payload):
    """Raise (before any money is spent) when today's estimated spend would
    exceed the daily cap. Cap <= 0 disables the brake."""
    est = estimate_cost(endpoint, payload)
    cap = spend_cap_usd()
    if cap <= 0:
        return est
    spent = spent_today()
    if spent + est > cap:
        raise RuntimeError(
            f"ATLAS DAILY SPEND CAP: ${spent:.2f} spent today + ${est:.2f} for this "
            f"{payload.get('model')} call exceeds the ${cap:.2f}/day cap "
            f"(config/generators.json generators.atlas.daily_spend_cap_usd, or "
            f"ATLAS_SPEND_CAP_USD env). Refusing to submit.")
    return est


def submit(endpoint, payload, key):
    est = check_spend_cap(endpoint, payload)
    resp = _curl("POST", f"{BASE}/{endpoint}", key, body=payload)
    pid = (resp.get("data", {}) or {}).get("id") or resp.get("id")
    if not pid:
        raise RuntimeError(f"no prediction id in response: "
                           f"{json.dumps(resp)[:600]}")
    _record_spend(endpoint, payload, est)
    return pid, resp


def poll(pid, key, timeout=900, interval=4, verbose=True):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = _curl("GET", f"{BASE}/model/prediction/{pid}", key)
        data = r.get("data", r)
        status = (data.get("status") or "").lower()
        if status != last and verbose:
            print(f"  [{int(time.time()%100000)}] status={status}",
                  file=sys.stderr)
            last = status
        if status in ("completed", "succeeded", "success"):
            outs = data.get("outputs") or data.get("output") or []
            if isinstance(outs := outs, str):
                outs = [outs]
            if not outs:
                raise RuntimeError(f"completed but no outputs: "
                                   f"{json.dumps(data)[:600]}")
            return outs, data
        if status in ("failed", "error", "canceled"):
            raise RuntimeError(f"generation {status}: "
                               f"{json.dumps(data)[:800]}")
        time.sleep(interval)
    raise TimeoutError(f"prediction {pid} did not finish in {timeout}s")


def download(url, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run(["curl", "-sS", "--fail", "-o", out, "-H",
                    f"User-Agent: {UA}", url], check=True)
    return out


def gen_image(model, prompt, refs, out, aspect=None, size=None,
              resolution=None, key=None, **extra):
    key = key or load_key()
    payload = {"model": model, "prompt": prompt}
    if refs:
        payload["images"] = [as_ref(r) for r in refs]
    if aspect:
        payload["aspect_ratio"] = aspect
    if size:
        payload["size"] = size
    if resolution:
        payload["resolution"] = resolution
    payload.update(extra)
    pid, _ = submit("model/generateImage", payload, key)
    print(f"submitted image {pid} ({model})", file=sys.stderr)
    outs, data = poll(pid, key)
    download(outs[0], out)
    return out, outs, data


def gen_video(model, prompt, refs, out, ratio="16:9", resolution="720p",
              duration=5, generate_audio=False, watermark=False,
              ref_videos=None, key=None, **extra):
    key = key or load_key()
    payload = {"model": model, "prompt": prompt, "ratio": ratio,
               "resolution": resolution, "duration": duration,
               "generate_audio": generate_audio, "watermark": watermark}
    if refs:
        payload["reference_images"] = [as_ref(r) for r in refs]
    if ref_videos:
        # videos must be a web url / asset:// (base64 rejected) -> upload local files
        payload["reference_videos"] = [as_video_ref(r, key=key) for r in ref_videos]
    payload.update(extra)
    pid, _ = submit("model/generateVideo", payload, key)
    print(f"submitted video {pid} ({model})", file=sys.stderr)
    outs, data = poll(pid, key, timeout=1200)
    download(outs[0], out)
    return out, outs, data


def main():
    ap = argparse.ArgumentParser(description="Atlas Cloud generation client")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("image", help="image / edit / reference-to-image")
    pi.add_argument("--model", default="google/nano-banana-pro/edit")
    pi.add_argument("--prompt", required=True)
    pi.add_argument("--ref", action="append", default=[], help="reference image (repeatable)")
    pi.add_argument("--aspect", help="e.g. 4:5 9:16 16:9")
    pi.add_argument("--size", help="WIDTH*HEIGHT (seedream-style models)")
    pi.add_argument("--resolution", help="e.g. 2k 4k")
    pi.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="extra payload field, repeatable (e.g. --set quality=high)")
    pi.add_argument("--out", required=True)

    pv = sub.add_parser("video", help="reference/image-to-video")
    pv.add_argument("--model", default="bytedance/seedance-2.0/reference-to-video")
    pv.add_argument("--prompt", required=True)
    pv.add_argument("--ref", action="append", default=[], help="reference image (repeatable)")
    pv.add_argument("--ref-video", action="append", default=[], help="reference video (repeatable)")
    pv.add_argument("--ratio", default="16:9")
    pv.add_argument("--resolution", default="720p")
    pv.add_argument("--duration", type=int, default=5)
    pv.add_argument("--audio", action="store_true")
    pv.add_argument("--out", required=True)

    pm = sub.add_parser("models", help="dump model catalog")
    pm.add_argument("--type", help="filter: Image/Video/Text/Audio")
    pm.add_argument("--grep")

    a = ap.parse_args()

    def parse_set(pairs):
        d = {}
        for kv in pairs:
            k, _, v = kv.partition("=")
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            elif v.lstrip("-").isdigit():
                v = int(v)
            d[k] = v
        return d

    if a.cmd == "image":
        out, outs, _ = gen_image(a.model, a.prompt, a.ref, a.out,
                                 aspect=a.aspect, size=a.size,
                                 resolution=a.resolution, **parse_set(a.set))
        print(json.dumps({"out": out, "remote": outs}, indent=2))
    elif a.cmd == "video":
        out, outs, _ = gen_video(a.model, a.prompt, a.ref, a.out,
                                 ratio=a.ratio, resolution=a.resolution,
                                 duration=a.duration, generate_audio=a.audio,
                                 ref_videos=a.ref_video or None)
        print(json.dumps({"out": out, "remote": outs}, indent=2))
    elif a.cmd == "models":
        key = load_key()
        cat = _curl("GET", f"{BASE}/models", key)["data"]
        for r in cat:
            if a.type and r.get("type") != a.type:
                continue
            if a.grep and a.grep.lower() not in r["model"].lower():
                continue
            price = (r.get("price", {}).get("actual", {}) or {}).get("base_price", "?")
            print(f"{r['model']:55s} {r['type']:6s} ${price}")


if __name__ == "__main__":
    main()
