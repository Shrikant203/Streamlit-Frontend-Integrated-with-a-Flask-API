"""
e2e_ui_tests.py
===============
End-to-end test of the integrated application: a real browser (Playwright / Chromium) drives the Streamlit
frontend, which calls the real Flask API. Each step checks what the user sees against what the API returns,
and saves a screenshot. Writes tests/results/ui_test_results.json, screens/*.png and e2e_flask_server.log.

Requires: pip install playwright && playwright install chromium
Run:      python tests/e2e_ui_tests.py
"""
import asyncio
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time

import numpy as np
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results"); SCR = os.path.join(RES, "screens")
os.makedirs(SCR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "frontend"))
from api_client import DigitAPIClient                       # noqa: E402
from preprocessing import smart_preprocess_to_8x8           # noqa: E402
from sklearn.datasets import load_digits                    # noqa: E402

CHROME = os.environ.get("CHROME_PATH", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
API, UI_PORT = "http://127.0.0.1:5000", 8610
UI = f"http://localhost:{UI_PORT}"
PHOTO = os.path.join(HERE, "sample_photo.png")
procs = {}
results = []


def start_flask():
    log = open(os.path.join(RES, "e2e_flask_server.log"), "a")
    procs["flask"] = subprocess.Popen([sys.executable, "app.py"], cwd=os.path.join(ROOT, "flask_api"), stdout=log,
                                      stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    for _ in range(100):
        if DigitAPIClient(API, 1).health().ok:
            return
        time.sleep(0.2)
    raise RuntimeError("Flask did not start")


def stop(name):
    p = procs.pop(name, None)
    if p:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=5)
        except Exception:
            pass


def start_streamlit():
    env = dict(os.environ, DIGIT_API_URL=API)
    procs["st"] = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.headless", "true",
                                    "--server.port", str(UI_PORT), "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env,
                                   stdout=open(os.path.join(RES, "streamlit.log"), "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    time.sleep(6)


def record(tid, name, expected, ok, actual):
    results.append({"id": tid, "name": name, "expected": expected, "actual": actual, "passed": bool(ok)})
    print(f"[{'PASS' if ok else 'FAIL'}] {tid} {name} -> {actual}")


async def settle(pg, ms=1800):
    await pg.wait_for_timeout(ms)


async def to_top(pg):
    await pg.mouse.move(2, 2)
    await pg.evaluate("""() => { for (const s of ['[data-testid="stMain"]','section.main','.stApp']) { const e=document.querySelector(s); if(e) e.scrollTop=0; } window.scrollTo(0,0); }""")
    await pg.wait_for_timeout(250)


async def fit(pg, width=1280):
    await pg.set_viewport_size({"width": width, "height": 1000}); await pg.wait_for_timeout(700)
    await to_top(pg)
    h = await pg.evaluate("""() => { const m=document.querySelector('[data-testid="stMainBlockContainer"]'); return Math.ceil(m.getBoundingClientRect().bottom + 90); }""")
    await pg.set_viewport_size({"width": width, "height": max(h, 900)})
    await pg.wait_for_timeout(900)


async def shot_main(pg, name):
    await fit(pg)
    await pg.locator('[data-testid="stMainBlockContainer"]').screenshot(path=os.path.join(SCR, name))


async def shot_full(pg, name, width=1280):
    await fit(pg, width)
    await pg.screenshot(path=os.path.join(SCR, name))


def status_line(side):
    m = re.findall(r"API (?:online|offline)[^\n]*|model loaded[^\n]*|Cannot reach[^\n]*", side)
    return " · ".join(x.strip() for x in m)


async def visible_metrics(pg):
    return await pg.evaluate("""() => [...document.querySelectorAll('[data-testid="stMetric"]')].filter(e => e.offsetParent !== null).map(e => e.innerText.replace(/\\n+/g, ' | '))""")


async def pick_select(pg, container_idx, option):
    box = pg.locator('[data-testid="stSelectbox"]:visible').nth(container_idx)
    await box.click()
    await pg.get_by_role("option", name=option, exact=True).click()
    await settle(pg, 1200)


async def stroke(pg, box, pts):
    px = lambda p: (box["x"] + p[0] * box["width"], box["y"] + p[1] * box["height"])
    await pg.mouse.move(*px(pts[0])); await pg.mouse.down()
    for p in pts[1:]:
        await pg.mouse.move(*px(p), steps=2)
    await pg.mouse.up()


def arc(cx, cy, rx, ry, a0, a1, n=50, seed=0):
    r = random.Random(seed)
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)) + r.uniform(-.004, .004),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n)) + r.uniform(-.004, .004)) for i in range(n + 1)]


async def run():
    digits = load_digits(); IM, LB = digits.images, digits.target
    direct = DigitAPIClient(API, 5)
    start_flask(); start_streamlit()
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=2)
        await pg.goto(UI, wait_until="networkidle"); await settle(pg, 3000)
        try:
            await steps(pg, digits, IM, LB, direct)
        except Exception:
            await pg.screenshot(path=os.path.join(SCR, "debug_failure.png"))
            raise
        await b.close()
    json.dump(results, open(os.path.join(RES, "ui_test_results.json"), "w"), indent=2)
    print(f"\n{sum(r['passed'] for r in results)}/{len(results)} UI checks passed")


async def steps(pg, digits, IM, LB, direct):
    if True:

        # U01 connection status
        side = await pg.locator('[data-testid="stSidebar"]').inner_text()
        record("U01", "App loads and the sidebar shows the Flask API as online", "'API online' in sidebar", "API online" in side, status_line(side))
        await shot_full(pg, "u01_home_online.png")

        # U02 sample prediction (digit 7) via the UI == direct API call
        await pick_select(pg, 0, "7")
        idx = int(np.where(LB == 7)[0][0])
        await pg.get_by_role("button", name="🚀 Send to Flask API").click()
        await pg.get_by_text("Result from the Flask API").wait_for(timeout=15000); await settle(pg)
        ui_digit = int(await pg.locator(".result-digit").first.inner_text())
        api_digit = direct.predict(IM[idx].astype(float)).data["predicted_digit"]
        await shot_full(pg, "u02_full.png")
        await pg.get_by_text("Show the raw request and response").click(); await settle(pg, 800)
        record("U02", "Sample digit 7: digit shown in the UI equals the API's answer", f"UI = API = 7", ui_digit == api_digit == 7, f"UI shows {ui_digit}, API returns {api_digit}")
        await shot_main(pg, "u02_sample_result.png")

        # U03 test-time augmentation uses one batch call
        await pg.get_by_text("Test-time augmentation").first.click(); await settle(pg, 1000)
        await pg.get_by_role("button", name="🚀 Send to Flask API").click(); await settle(pg, 2500)
        mets = " ".join(await visible_metrics(pg))
        record("U03", "Test-time augmentation sends 11 variants in one /predict/batch request", "endpoint '/predict/batch'", "/predict/batch" in mets, mets[:110])
        await shot_main(pg, "u03_tta_result.png")
        await pg.get_by_text("Test-time augmentation").first.click(); await settle(pg, 800)

        # U04 drawing
        await pg.get_by_text("Draw", exact=True).first.click(); await settle(pg, 3500)
        cv = pg.locator("canvas.upper-canvas").first
        box = await cv.bounding_box()
        await stroke(pg, box, arc(0.47, 0.30, 0.20, 0.17, -150, 90, seed=1)); await stroke(pg, box, arc(0.47, 0.64, 0.23, 0.19, -90, 150, seed=2))
        await settle(pg, 3000)
        await pg.get_by_role("button", name="🚀 Send to Flask API").click(); await pg.wait_for_timeout(3500)
        ui_digit = int(await pg.locator(".result-digit").first.inner_text())
        record("U04", "Hand-drawn '3' is sent to the API and the prediction is shown", "digit 3", ui_digit == 3, f"UI shows {ui_digit}")
        await shot_main(pg, "u04_draw_result.png")

        # U05 upload
        await pg.get_by_text("Upload photo", exact=True).first.click(); await settle(pg, 1500)
        async with pg.expect_file_chooser(timeout=15000) as fc:
            await pg.get_by_test_id("stFileUploaderDropzone").locator("button").first.click()
        await (await fc.value).set_files(PHOTO); await settle(pg, 3000)
        await pg.get_by_role("button", name="🚀 Send to Flask API").click(); await pg.wait_for_timeout(3500)
        ui_digit = int(await pg.locator(".result-digit").first.inner_text())
        record("U05", "Uploaded photo of a '2' is preprocessed, sent and classified", "digit 2", ui_digit == 2, f"UI shows {ui_digit}")
        await shot_main(pg, "u05_upload_result.png")

        # U06 batch (raw, then preprocessed)
        await pg.get_by_role("tab", name=re.compile("Batch test")).click(); await settle(pg, 1200)
        await pg.get_by_role("button", name="▶ Run batch").click(); await settle(pg, 3000)
        m_raw = await visible_metrics(pg)
        acc_raw = float(re.search(r"Accuracy \| ([\d.]+)%", m_raw[0]).group(1))
        await shot_main(pg, "u06_batch_raw.png")
        await pg.get_by_text("Crop → scale → center first").filter(visible=True).first.click(); await settle(pg, 800)
        await pg.get_by_role("button", name="▶ Run batch").click(); await settle(pg, 3000)
        m_pre = await visible_metrics(pg)
        acc_pre = float(re.search(r"Accuracy \| ([\d.]+)%", m_pre[0]).group(1))
        sel = np.random.RandomState(1).choice(np.arange(len(IM)), size=50, replace=False)
        direct_res = direct.predict_batch([IM[i].astype(float) for i in sel]).data["results"]
        exp_raw = round(float(np.mean([r["predicted_digit"] == LB[i] for r, i in zip(direct_res, sel)])) * 100, 1)
        record("U06", "Batch tab: 50 images in one request; UI accuracy equals a direct API call", f"raw accuracy = {exp_raw}%", abs(acc_raw - exp_raw) < 0.05,
               f"UI raw {acc_raw}% | UI preprocessed {acc_pre}% | direct raw {exp_raw}%")
        await shot_main(pg, "u06_batch_pre.png")

        # U07 API explorer: handled errors
        await pg.get_by_role("tab", name=re.compile("API explorer")).click(); await settle(pg, 1200)
        cases = [("POST /predict  — wrong size (3 values)", "u07a_explorer_400.png", 400),
                 ("POST /predict/batch  — one bad image in the batch", "u07b_explorer_batch.png", 200),
                 ("GET /nonexistent  (expect 404)", "u07c_explorer_404.png", 404)]
        for preset, fname, code in cases:
            await pick_select(pg, 0, preset)
            await pg.get_by_role("button", name="Send request").click(); await settle(pg, 1800)
            txt = await pg.locator('[data-testid="stMainBlockContainer"]').inner_text()
            record("U07", f"API explorer: '{preset.split('  ')[0]} {preset.split('  ')[-1][:34]}' shows HTTP {code}", f"HTTP {code}", f"HTTP {code}" in txt, f"page shows 'HTTP {code}'" if f"HTTP {code}" in txt else "status not found")
            await shot_main(pg, fname)

        # U08 request log
        await pg.get_by_role("tab", name=re.compile("Request log")).click(); await settle(pg, 1500)
        mets = await visible_metrics(pg)
        n_req = int(re.search(r"Requests \| (\d+)", mets[0]).group(1))
        record("U08", "Request log records every call made by the UI", "log has >= 8 requests", n_req >= 8, f"{n_req} requests logged; {' / '.join(mets)}"[:150])
        await shot_main(pg, "u08_request_log.png")

        # U09 API goes offline, then recovers
        stop("flask"); await settle(pg, 500)
        await pg.goto(UI, wait_until="networkidle"); await settle(pg, 3500)     # a fresh visit while the API is down
        side = await pg.locator('[data-testid="stSidebar"]').inner_text()
        disabled = await pg.get_by_role("button", name="🚀 Send to Flask API").is_disabled()
        record("U09", "API stopped: UI shows 'API offline', explains how to start it, disables Send", "offline banner + disabled button", "API offline" in side and disabled, f"sidebar: {status_line(side)} | Send button disabled = {disabled}")
        await shot_full(pg, "u09_offline.png", width=1000)
        start_flask()
        await pg.get_by_role("button", name="Re-check connection").click(); await settle(pg, 3000)
        side = await pg.locator('[data-testid="stSidebar"]').inner_text()
        await pg.get_by_role("button", name="🚀 Send to Flask API").click(); await pg.wait_for_timeout(3000)
        ok = "API online" in side and await pg.locator(".result-digit").count() > 0
        record("U10", "API restarted: UI reconnects and predictions work again", "'API online' + a result", ok, f"sidebar: {status_line(side)}; a new result is displayed")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    finally:
        stop("st"); stop("flask")
