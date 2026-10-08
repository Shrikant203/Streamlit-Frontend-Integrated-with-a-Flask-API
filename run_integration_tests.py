"""
run_integration_tests.py
========================
Task 8 test report generator. Starts the REAL Flask API (flask_api/app.py, unchanged from Task 4) in a
subprocess and exercises it through the same DigitAPIClient the Streamlit frontend uses.

Covers: functional behaviour, error handling, faithfulness (API == local model), accuracy through the API,
latency, concurrency and crash recovery. Writes tests/results/test_results.json and flask_server.log.

Run:  python tests/run_integration_tests.py
"""
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.datasets import load_digits

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "frontend"))
from api_client import DigitAPIClient                                   # noqa: E402
from preprocessing import smart_preprocess_to_8x8, tta_variants         # noqa: E402

URL = "http://127.0.0.1:5000"
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

_server = None


def start_server():
    global _server
    log = open(os.path.join(RESULTS, "flask_server.log"), "a")
    _server = subprocess.Popen([sys.executable, "app.py"], cwd=os.path.join(ROOT, "flask_api"),
                               stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    t0 = time.time()
    while time.time() - t0 < 20:
        if DigitAPIClient(URL, 1).health().ok:
            return time.time() - t0
        time.sleep(0.2)
    raise RuntimeError("Flask API did not start")


def stop_server():
    global _server
    if _server is not None:
        try:
            os.killpg(os.getpgid(_server.pid), signal.SIGTERM)
            _server.wait(timeout=5)
        except Exception:
            pass
        _server = None


# ---------------------------------------------------------------------------
digits = load_digits()
IMAGES, LABELS = digits.images, digits.target
perm = np.random.default_rng(42).permutation(len(IMAGES))
HELD_OUT = perm[int(len(IMAGES) * 0.85):]                   # 270 images, same split as Task 7

spec = importlib.util.spec_from_file_location("flask_app", os.path.join(ROOT, "flask_api", "app.py"))
flask_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flask_app)                           # gives us the API's own model + parse_image for comparison
local_model = flask_app.model

tests, metrics = [], {}
client = DigitAPIClient(URL, 5)


def run(tid, group, name, expected, fn):
    t0 = time.perf_counter()
    try:
        ok, actual = fn()
    except Exception as e:                                   # a crashing test is a failed test
        ok, actual = False, f"exception: {e!r}"
    ms = (time.perf_counter() - t0) * 1000
    tests.append({"id": tid, "group": group, "name": name, "expected": expected, "actual": actual,
                  "passed": bool(ok), "ms": round(ms, 1)})
    print(f"[{'PASS' if ok else 'FAIL'}] {tid} {name}  ->  {actual}")


def first_of(d):
    return int(np.where(LABELS == d)[0][0])


def api_predict_many(indices, preprocess=False):
    preds, probs = [], []
    for s in range(0, len(indices), 100):
        chunk = indices[s:s + 100]
        imgs = [smart_preprocess_to_8x8(np.clip(IMAGES[i], 0, 16) / 16 * 255) if preprocess else IMAGES[i].astype(float) for i in chunk]
        r = client.predict_batch(imgs)
        assert r.ok, r.message
        for res in r.data["results"]:
            preds.append(res["predicted_digit"]); probs.append([res["probabilities"][str(d)] for d in range(10)])
    return np.array(preds), np.array(probs)


def main():
    boot = start_server()
    metrics["server_start_s"] = round(boot, 2)
    print(f"Flask API started in {boot:.2f} s\n")
    zero = [0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0, 0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0,
            0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0, 0, 2, 14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0]

    # ---------------- A. connection and information
    run("T01", "Connection", "GET / returns the API description", "200 + 3 endpoints listed",
        lambda: ((r := client.info()).ok and len(r.data["endpoints"]) == 3, f"HTTP {r.status}, endpoints={len(r.data['endpoints'])}"))
    run("T02", "Connection", "GET /health reports the model as loaded", "200, model_loaded = true",
        lambda: ((r := client.health()).ok and r.data["model_loaded"] is True, f"HTTP {r.status}, {r.data}"))

    # ---------------- B. predictions
    def t03():
        i = first_of(3); r = client.predict(IMAGES[i])
        s = sum(r.data["probabilities"].values())
        return r.ok and r.data["predicted_digit"] == 3 and abs(s - 1) < 1e-3, f"HTTP {r.status}, digit {r.data['predicted_digit']} @ {r.data['confidence']}, sum(p)={s:.4f}"
    run("T03", "Prediction", "Single raw image (true 3) via POST /predict", "digit 3, probabilities sum to 1", t03)

    def t04():
        i = first_of(7); px = smart_preprocess_to_8x8(np.clip(IMAGES[i], 0, 16) / 16 * 255); r = client.predict(px)
        return r.ok and r.data["predicted_digit"] == 7, f"digit {r.data['predicted_digit']} @ {r.data['confidence']}"
    run("T04", "Prediction", "Frontend-preprocessed image (true 7) via POST /predict", "digit 7", t04)

    def t05():
        r16 = client.custom("POST", "/predict", json.dumps({"image": zero}))
        r01 = client.custom("POST", "/predict", json.dumps({"image": [v / 16 for v in zero]}))
        same = r16.data["predicted_digit"] == r01.data["predicted_digit"] == 0
        return same and r16.ok and r01.ok, f"0-16 scale -> {r16.data['predicted_digit']}, 0-1 scale -> {r01.data['predicted_digit']}"
    run("T05", "Prediction", "0-16 and 0-1 pixel scales give the same digit", "both predict 0", t05)

    run("T06", "Prediction", "Nested 8x8 list is accepted", "200, digit 0",
        lambda: ((r := client.custom("POST", "/predict", json.dumps({"image": [zero[k:k + 8] for k in range(0, 64, 8)]}))).ok and r.data["predicted_digit"] == 0,
                 f"HTTP {r.status}, digit {r.data.get('predicted_digit')}"))

    def t07():
        idx = [first_of(3), first_of(7), first_of(8)]; r = client.predict_batch([IMAGES[i] for i in idx])
        got = [x["predicted_digit"] for x in r.data["results"]]
        return r.ok and got == [3, 7, 8], f"HTTP {r.status}, predicted {got}, count={r.data['count']}"
    run("T07", "Prediction", "Batch of 3 real digits (3, 7, 8)", "[3, 7, 8]", t07)

    def t08():
        variants = tta_variants(IMAGES[first_of(8)].astype(float)); r = client.predict_batch(variants)
        probs = np.mean([[x["probabilities"][str(d)] for d in range(10)] for x in r.data["results"]], axis=0)
        return r.ok and r.data["count"] == 11 and abs(probs.sum() - 1) < 1e-3 and int(probs.argmax()) == 8, \
            f"{r.data['count']} variants, averaged digit {int(probs.argmax())} @ {probs.max():.4f}"
    run("T08", "Prediction", "Test-time augmentation: 11 variants in one batch call", "11 results, averaged digit 8", t08)

    def t09():
        r = client.predict_batch([IMAGES[i] for i in range(100)])
        return r.ok and r.data["count"] == 100, f"HTTP {r.status}, count={r.data['count']}"
    run("T09", "Prediction", "Batch at the limit (100 images)", "200, 100 results", t09)

    # ---------------- C. error handling
    def err(method, path, body, status, contains):
        r = client.custom(method, path, body)
        msg = (r.data or {}).get("error", "")
        return r.status == status and contains in msg, f"HTTP {r.status}: {msg}"
    run("T10", "Errors", "Missing 'image' field", "400", lambda: err("POST", "/predict", "{}", 400, "Missing 'image'"))
    run("T11", "Errors", "Wrong-size image (3 values)", "400", lambda: err("POST", "/predict", '{"image":[1,2,3]}', 400, "got 3 values"))
    run("T12", "Errors", "Invalid JSON body", "400", lambda: err("POST", "/predict", '{"image": [1, 2, ', 400, "valid JSON"))
    run("T13", "Errors", "Pixel value above range (99)", "400", lambda: err("POST", "/predict", json.dumps({"image": [99] + zero[1:]}), 400, "within [0, 16]"))
    run("T14", "Errors", "Negative pixel value", "400", lambda: err("POST", "/predict", json.dumps({"image": [-1] + zero[1:]}), 400, "within"))
    run("T15", "Errors", "Non-numeric pixel values", "400", lambda: err("POST", "/predict", json.dumps({"image": ["a"] * 64}), 400, ""))
    run("T16", "Errors", "Unknown endpoint", "404", lambda: err("GET", "/nonexistent", None, 404, "not found"))
    run("T17", "Errors", "Wrong HTTP method (GET /predict)", "405", lambda: err("GET", "/predict", None, 405, "Method not allowed"))
    run("T18", "Errors", "Empty batch list", "400", lambda: err("POST", "/predict/batch", '{"images": []}', 400, "non-empty"))
    run("T19", "Errors", "Batch larger than 100", "400", lambda: err("POST", "/predict/batch", json.dumps({"images": [zero] * 101}), 400, "limited to 100"))

    def t20():
        r = client.custom("POST", "/predict/batch", json.dumps({"images": [zero, [1, 2, 3], zero]}))
        res = r.data["results"]
        return r.status == 200 and "error" in res[1] and "predicted_digit" in res[0] and "predicted_digit" in res[2], \
            f"HTTP {r.status}: entries -> {['ok' if 'predicted_digit' in x else 'error' for x in res]}"
    run("T20", "Errors", "One bad image does not fail the whole batch", "200; entries [ok, error, ok]", t20)

    # ---------------- D. faithfulness and accuracy through the API
    def t21():
        allidx = np.arange(len(IMAGES)); preds, probs = api_predict_many(allidx)
        x = np.stack([flask_app.parse_image(IMAGES[i].flatten().tolist())[0] for i in allidx])
        lp = local_model.predict_proba(x)
        same = int((lp.argmax(1) == preds).sum()); maxdiff = float(np.abs(lp - probs).max())
        metrics["consistency"] = {"images": len(allidx), "same_prediction": same, "max_abs_prob_diff": maxdiff}
        return same == len(allidx) and maxdiff < 1e-4, f"{same}/{len(allidx)} identical predictions, max |p_api - p_local| = {maxdiff:.1e}"
    run("T21", "Faithfulness", "API output equals the local model on all 1,797 images", "identical predictions, diff < 1e-4", t21)

    def t22():
        out = {}
        for name, idx in (("held_out_270", HELD_OUT), ("all_1797", np.arange(len(IMAGES)))):
            for pre in (False, True):
                p, _ = api_predict_many(idx, pre)
                out[f"{name}_{'preprocessed' if pre else 'raw'}"] = round(float((p == LABELS[idx]).mean()) * 100, 2)
        metrics["accuracy_via_api"] = out
        return out["held_out_270_preprocessed"] > 95 and out["held_out_270_raw"] > 90, json.dumps(out)
    run("T22", "Accuracy", "Accuracy measured through the HTTP API", "> 90 % raw, > 95 % preprocessed (held-out)", t22)

    # ---------------- E. latency, throughput, concurrency
    def t23():
        sel = np.random.default_rng(0).choice(HELD_OUT, 200)
        lat = []
        for i in sel:
            r = client.predict(IMAGES[i]); assert r.ok; lat.append(r.latency_ms)
        a = np.array(lat)
        m = {"n": 200, "mean_ms": round(a.mean(), 1), "median_ms": round(float(np.median(a)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
             "p99_ms": round(float(np.percentile(a, 99)), 1), "max_ms": round(a.max(), 1), "min_ms": round(a.min(), 1)}
        metrics["latency_single"] = m
        return m["p95_ms"] < 250, f"median {m['median_ms']} ms, p95 {m['p95_ms']} ms, max {m['max_ms']} ms"
    run("T23", "Performance", "200 sequential POST /predict calls", "p95 round-trip < 250 ms", t23)

    def t24():
        times = []
        for _ in range(5):
            r = client.predict_batch([IMAGES[i] for i in range(100)]); assert r.ok; times.append(r.latency_ms)
        m = {"batch_size": 100, "median_ms": round(float(np.median(times)), 1), "per_image_ms": round(float(np.median(times)) / 100, 2)}
        metrics["latency_batch"] = m
        return True, f"100 images in {m['median_ms']} ms ({m['per_image_ms']} ms per image)"
    run("T24", "Performance", "POST /predict/batch with 100 images (median of 5)", "completes; per-image cost reported", t24)

    def t25():
        sel = np.random.default_rng(1).choice(HELD_OUT, 200)
        def one(i):
            c = DigitAPIClient(URL, 10); r = c.predict(IMAGES[i]); return r.ok and r.data["predicted_digit"] is not None, r.latency_ms
        t0 = time.perf_counter()
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, sel))
        wall = time.perf_counter() - t0
        okn = sum(1 for o, _ in res if o)
        metrics["concurrency"] = {"workers": 8, "requests": 200, "succeeded": okn, "wall_s": round(wall, 2), "throughput_rps": round(200 / wall, 1)}
        return okn == 200, f"{okn}/200 succeeded with 8 parallel clients, {200/wall:.1f} requests/s"
    run("T25", "Performance", "200 requests from 8 parallel clients", "all 200 succeed", t25)

    # ---------------- F. resilience
    def t26():
        stop_server(); time.sleep(0.5)
        h, p = client.health(), client.predict(IMAGES[0])
        return h.unreachable and p.unreachable and h.error and p.error, f"health -> '{h.message}' (no exception raised)"
    run("T26", "Resilience", "API stopped: client reports an error instead of crashing", "unreachable result with message", t26)

    def t27():
        t0 = time.time(); start_server(); rec = time.time() - t0
        r = client.predict(IMAGES[first_of(3)]); metrics["recovery_s"] = round(rec, 2)
        return r.ok and r.data["predicted_digit"] == 3, f"API restarted; first prediction OK after {rec:.2f} s"
    run("T27", "Resilience", "API restarted: the same client works again", "prediction succeeds", t27)

    passed = sum(t["passed"] for t in tests)
    metrics["passed"], metrics["total"] = passed, len(tests)
    json.dump({"tests": tests, "metrics": metrics}, open(os.path.join(RESULTS, "test_results.json"), "w"), indent=2)
    print(f"\n{passed}/{len(tests)} tests passed")
    print(json.dumps(metrics, indent=2))
    stop_server()


if __name__ == "__main__":
    try:
        main()
    finally:
        stop_server()
