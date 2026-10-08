"""
streamlit_app.py
================
Task 8: Integrating a Streamlit frontend with the Flask API (Task 4).

The frontend contains NO model code. Every prediction is made by sending a JSON
request to the Flask backend over HTTP and displaying the JSON that comes back:

    browser  <->  Streamlit (this file)  --HTTP/JSON-->  Flask API  -->  CNN model

Run the backend first (cd flask_api && python app.py), then:
    streamlit run frontend/streamlit_app.py
The API address can be changed in the sidebar or with the DIGIT_API_URL variable.
"""
import json
import os
import time
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.datasets import load_digits

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except Exception:                      # component missing or incompatible
    CANVAS_OK = False

from api_client import DigitAPIClient
from preprocessing import crop_to_ink, pixels_to_display_image, smart_preprocess_to_8x8, tta_variants

DEFAULT_URL = os.environ.get("DIGIT_API_URL", "http://127.0.0.1:5000")
DIGITS = [str(i) for i in range(10)]

st.set_page_config(page_title="Digit Classifier · Streamlit + Flask", page_icon="🔢", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1250px; }
.hero { background: linear-gradient(180deg,#17181c 0%,#101114 100%); border: 1px solid rgba(255,255,255,.08);
        border-top: 3px solid #FF4B4B; border-radius: 10px; padding: 1.4rem 2rem; margin-bottom: 1.2rem; }
.hero .eyebrow { font-family: monospace; font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; color: #9CA3AF; }
.hero h1 { margin: .5rem 0 0 0 !important; color: #F5F5F7; font-size: 2rem !important; font-weight: 800 !important; }
.hero p { margin: .5rem 0 0 0; color: #c9ccd1; max-width: 820px; line-height: 1.55; }
.result-card { border-radius: 18px; padding: 1.4rem 1.2rem; text-align: center; border: 2px solid var(--b); background: var(--bg); }
.result-digit { font-size: 4.2rem; font-weight: 800; line-height: 1; margin: .2rem 0; }
.result-label { font-size: .9rem; opacity: .75; text-transform: uppercase; letter-spacing: .06em; }
.pill { display: inline-block; padding: .3rem .9rem; border-radius: 999px; font-weight: 700; margin-top: .5rem; }
.step { display: flex; align-items: center; gap: .6rem; margin-bottom: .45rem; }
.badge { width: 28px; height: 28px; border-radius: 50%; background: #FF4B4B; color: #fff; font-weight: 800;
         display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }
.footer { text-align: center; opacity: .55; font-size: .85rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Request log + API client (one client per rerun, calls are recorded in session state)
# ---------------------------------------------------------------------------
st.session_state.setdefault("api_log", [])


def record_call(r):
    st.session_state.api_log.append({
        "time": datetime.now().strftime("%H:%M:%S"), "method": r.method, "endpoint": r.path,
        "status": r.status if r.status is not None else "no response",
        "latency_ms": round(r.latency_ms, 1), "result": "OK" if r.ok else r.message,
    })


@st.cache_data
def load_dataset():
    d = load_digits()
    return d.images, d.target


images, labels = load_dataset()

with st.sidebar:
    st.markdown("### 🔌 Flask API connection")
    api_url = st.text_input("API base URL", DEFAULT_URL)
    timeout = st.slider("Request timeout (s)", 1, 15, 5)
    st.button("Re-check connection", width="stretch")
    client = DigitAPIClient(api_url, timeout, on_call=record_call)
    health = client.health(record=False)
    API_ONLINE = bool(health.ok and isinstance(health.data, dict) and health.data.get("model_loaded"))
    if API_ONLINE:
        st.success(f"**API online** ✓\n\nmodel loaded · {health.latency_ms:.0f} ms")
    else:
        st.error(f"**API offline** ✗\n\n{health.message}")
        st.code("cd flask_api\npython app.py", language="bash")
    st.divider()
    log = st.session_state.api_log
    st.caption(f"Requests sent this session: **{len(log)}**")
    if log:
        ok = sum(1 for e in log if e["result"] == "OK")
        st.caption(f"Successful: **{ok}** · mean latency: **{np.mean([e['latency_ms'] for e in log]):.1f} ms**")

st.markdown("""
<div class="hero"><div class="eyebrow">Task 8 · Streamlit frontend + Flask backend</div>
<h1>Digit Classifier</h1>
<p>Draw, upload or pick a digit. The Streamlit frontend sends it as JSON to a separate Flask API,
which runs the CNN and returns the prediction in real time.</p></div>
""", unsafe_allow_html=True)


def steps(*items):
    st.markdown("".join(f'<div class="step"><span class="badge">{i}</span><span>{t}</span></div>'
                        for i, t in enumerate(items, 1)), unsafe_allow_html=True)


def confidence_style(c):
    if c >= 0.8:
        return "#21C55D", "High confidence"
    if c >= 0.5:
        return "#F59E0B", "Moderate confidence"
    return "#EF4444", "Low confidence"


# ---------------------------------------------------------------------------
# Sending a prediction through the API
# ---------------------------------------------------------------------------
def predict_via_api(pixels, use_tta):
    """Returns a dict describing the outcome, or {'error': ...} if the API call failed."""
    if use_tta:
        variants = tta_variants(pixels)
        r = client.predict_batch(variants)            # one POST /predict/batch with 11 images
        if not r.ok:
            return {"error": r.message, "result": r}
        rows = [x for x in r.data["results"] if "error" not in x]
        if not rows:
            return {"error": "The API rejected every variant.", "result": r}
        probs = np.mean([[x["probabilities"][d] for d in DIGITS] for x in rows], axis=0)
        mode = f"POST /predict/batch  ({len(variants)} test-time-augmentation variants averaged)"
    else:
        r = client.predict(pixels)                     # one POST /predict
        if not r.ok:
            return {"error": r.message, "result": r}
        probs = np.array([r.data["probabilities"][d] for d in DIGITS])
        mode = "POST /predict"
    pred = int(np.argmax(probs))
    return {"pred": pred, "conf": float(probs[pred]), "probs": probs, "mode": mode, "result": r, "pixels": pixels}


def render_result(out, true_label=None):
    if "error" in out:
        st.error(f"The API call failed. {out['error']}")
        return
    r, probs, pred, conf = out["result"], out["probs"], out["pred"], out["conf"]
    color, word = confidence_style(conf)
    st.markdown("#### Result from the Flask API")
    c1, c2, c3 = st.columns([1, 1.1, 1.3])
    with c1:
        st.image(pixels_to_display_image(out["pixels"], 220), caption="8×8 image sent to the API", width="stretch")
    with c2:
        verdict = ""
        if true_label is not None:
            verdict = f'<div style="margin-top:.5rem">{"✅ correct" if pred == true_label else "❌ wrong"} (true label {true_label})</div>'
        st.markdown(
            f'<div class="result-card" style="--b:{color}55;--bg:{color}14"><div class="result-label">Predicted digit</div>'
            f'<div class="result-digit" style="color:{color}">{pred}</div>'
            f'<span class="pill" style="background:{color}22;color:{color}">{conf*100:.1f}% · {word}</span>{verdict}</div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown("**Top 3 guesses**")
        for rank, d in enumerate(np.argsort(probs)[::-1][:3]):
            st.markdown(f"{['🥇', '🥈', '🥉'][rank]} **{d}** — {probs[d]*100:.1f}%")
            st.progress(float(min(max(probs[d], 0), 1)))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Round-trip latency", f"{r.latency_ms:.1f} ms")
    m2.metric("HTTP status", r.status)
    m3.metric("Endpoint", out["mode"].split("  ")[0].replace("POST ", ""))
    m4.metric("Payload sent", f"{len(json.dumps(r.request_body))/1024:.1f} KB")
    df = pd.DataFrame({"Digit": DIGITS, "Probability": probs, "Top": [int(i) == pred for i in range(10)]})
    st.altair_chart(
        alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Digit:N", sort=None), y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.Top, alt.value("#FF4B4B"), alt.value("#5B7FDE88")),
            tooltip=["Digit", alt.Tooltip("Probability:Q", format=".1%")]).properties(height=220),
        width="stretch")
    with st.expander("Show the raw request and response (JSON)"):
        a, b = st.columns(2)
        body = r.request_body
        shown = body if len(json.dumps(body)) < 900 else {k: (f"[{len(v)} images × 64 values]" if k == "images" else v) for k, v in body.items()}
        a.markdown(f"**Request** — `{out['mode']}`")
        a.code(json.dumps(shown), language="json", wrap_lines=True)
        resp = r.data
        if isinstance(resp, dict) and "results" in resp:
            resp = {"count": resp["count"], "results": f"[{len(resp['results'])} predictions]  first: {resp['results'][0]}"}
        b.markdown(f"**Response** — HTTP {r.status}")
        b.json(resp)


# ---------------------------------------------------------------------------
tab_predict, tab_batch, tab_explorer, tab_log = st.tabs(
    ["🧪 Predict", "📦 Batch test", "🛠️ API explorer", "📜 Request log"])

# ======================== Predict ========================
with tab_predict:
    steps("Choose how to provide a digit.", "Press <b>Send to Flask API</b> — the frontend posts JSON and shows the live reply.")
    mode = st.radio("Input", ["Sample digit", "Draw", "Upload photo"], horizontal=True, label_visibility="collapsed")
    use_tta = st.checkbox("Test-time augmentation (sends 11 variants in one /predict/batch call and averages them)")
    pixels, true_label = None, None

    if mode == "Sample digit":
        fcol, scol, icol = st.columns([1, 2, 1])
        dfilter = fcol.selectbox("Digit", ["All"] + DIGITS)
        cand = np.arange(len(images)) if dfilter == "All" else np.where(labels == int(dfilter))[0]
        pick = scol.slider("Sample number", 1, len(cand), 1)
        idx = int(cand[pick - 1])
        pre = icol.checkbox("Crop → scale → center first", value=False)
        true_label = int(labels[idx])
        raw = images[idx].astype(float)
        pixels = smart_preprocess_to_8x8(np.clip(raw, 0, 16) / 16 * 255) if pre else raw
        a, b = st.columns([1, 4])
        a.image(pixels_to_display_image(pixels, 190), width="stretch")
        b.markdown(f"Dataset index **{idx}** · true label **{true_label}**")

    elif mode == "Draw":
        if not CANVAS_OK:
            st.warning("The drawing canvas component is not available here. Use a sample or upload a photo.")
        else:
            left, right = st.columns([3, 2])
            with left:
                brush = st.slider("Brush size", 8, 40, 22)
                canvas = st_canvas(fill_color="white", stroke_width=brush, stroke_color="white", background_color="black",
                                   height=360, width=360, drawing_mode="freedraw", update_streamlit=True, return_image_data=True, key="canvas")
            if canvas.image_data is not None and canvas.image_data[:, :, :3].sum() > 0:
                gray = np.array(Image.fromarray(canvas.image_data.astype(np.uint8), mode="RGBA").convert("L"))
                pixels = smart_preprocess_to_8x8(gray)
            with right:
                st.markdown("#### 8×8 image that will be sent")
                if pixels is not None and pixels.sum() > 0:
                    st.image(pixels_to_display_image(pixels, 280), width="stretch")
                else:
                    pixels = None
                    st.info("Draw a digit to see the live preview.")

    else:
        up, opt = st.columns([2, 1])
        invert = opt.checkbox("Invert colors", value=True, help="ON for a dark digit on a light background.")
        file = up.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        if file is not None:
            gray = np.array(Image.open(file).convert("L")).astype(float)
            gray = 255 - gray if invert else gray
            crop = crop_to_ink(gray)
            pixels = smart_preprocess_to_8x8(gray)
            c1, c2, c3 = st.columns(3)
            c1.markdown("**Original**"); c1.image(file, width="stretch")
            c2.markdown("**Auto-cropped**")
            if crop is not None:
                c2.image(Image.fromarray(crop.astype(np.uint8)).resize((240, 240), Image.NEAREST), width="stretch")
            c3.markdown("**8×8 image to send**"); c3.image(pixels_to_display_image(pixels, 240), width="stretch")
            if pixels.sum() == 0:
                pixels = None
                st.warning("No digit found. Try the invert toggle or a clearer photo.")

    send = st.button("🚀 Send to Flask API", type="primary", disabled=(pixels is None or not API_ONLINE))
    if not API_ONLINE:
        st.warning("The Flask API is offline, so predictions are unavailable. Start it, then press *Re-check connection*.")
    if send and pixels is not None:
        with st.spinner("Waiting for the Flask API..."):
            st.session_state.last = (predict_via_api(pixels, use_tta), true_label)
    if st.session_state.get("last"):
        st.divider()
        render_result(*st.session_state.last)

# ======================== Batch ========================
with tab_batch:
    st.markdown("Send many dataset digits in **one** `POST /predict/batch` request and compare the API's answers with the true labels.")
    c1, c2, c3, c4 = st.columns(4)
    n = c1.slider("Images in the batch", 5, 100, 50)
    src = c2.selectbox("Digits", ["All"] + DIGITS, key="bsrc")
    seed = c3.number_input("Random seed", 0, 9999, 1)
    pre_b = c4.checkbox("Crop → scale → center first", value=False, key="bpre")
    if st.button("▶ Run batch", type="primary", disabled=not API_ONLINE):
        pool = np.arange(len(images)) if src == "All" else np.where(labels == int(src))[0]
        pick = np.random.RandomState(int(seed)).choice(pool, size=min(n, len(pool)), replace=False)
        batch = [smart_preprocess_to_8x8(np.clip(images[i], 0, 16) / 16 * 255) if pre_b else images[i].astype(float) for i in pick]
        with st.spinner("Sending batch..."):
            r = client.predict_batch(batch)
        if not r.ok:
            st.error(r.message)
        else:
            rows = []
            for i, res in zip(pick, r.data["results"]):
                if "error" in res:
                    rows.append({"Index": int(i), "True": int(labels[i]), "Predicted": None, "Confidence %": None, "Correct": False})
                else:
                    rows.append({"Index": int(i), "True": int(labels[i]), "Predicted": res["predicted_digit"],
                                 "Confidence %": round(res["confidence"] * 100, 1), "Correct": res["predicted_digit"] == int(labels[i])})
            st.session_state.batch = (pd.DataFrame(rows), r.latency_ms, r.status)
    if st.session_state.get("batch") is not None:
        df, ms, status = st.session_state.batch
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{df['Correct'].mean()*100:.1f}%", f"{int(df['Correct'].sum())} of {len(df)} correct")
        m2.metric("Mean confidence", f"{df['Confidence %'].mean():.1f}%")
        m3.metric("Batch round-trip", f"{ms:.0f} ms")
        m4.metric("Per image", f"{ms/len(df):.1f} ms", f"HTTP {status}")
        st.dataframe(df.style.apply(lambda row: ["background-color:#fde2e2" if not row["Correct"] else "" for _ in row], axis=1),
                     width="stretch", height=300, hide_index=True,
                     column_config={"Confidence %": st.column_config.NumberColumn(format="%.1f"),
                                    "Correct": st.column_config.CheckboxColumn(disabled=True)})

# ======================== API explorer ========================
ZERO = [0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0, 0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0,
        0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0, 0, 2, 14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0]
PRESETS = {
    "GET /  (API information)": ("GET", "/", ""),
    "GET /health": ("GET", "/health", ""),
    "POST /predict  — valid image": ("POST", "/predict", json.dumps({"image": ZERO})),
    "POST /predict  — missing 'image' field": ("POST", "/predict", "{}"),
    "POST /predict  — wrong size (3 values)": ("POST", "/predict", json.dumps({"image": [1, 2, 3]})),
    "POST /predict  — pixel value out of range (99)": ("POST", "/predict", json.dumps({"image": [99] + ZERO[1:]})),
    "POST /predict  — invalid JSON": ("POST", "/predict", '{"image": [1, 2, '),
    "POST /predict/batch  — two valid images": ("POST", "/predict/batch", json.dumps({"images": [ZERO, ZERO]})),
    "POST /predict/batch  — one bad image in the batch": ("POST", "/predict/batch", json.dumps({"images": [ZERO, [1, 2, 3]]})),
    "POST /predict/batch  — empty list": ("POST", "/predict/batch", json.dumps({"images": []})),
    "GET /nonexistent  (expect 404)": ("GET", "/nonexistent", ""),
    "GET /predict  (wrong method, expect 405)": ("GET", "/predict", ""),
}
with tab_explorer:
    st.markdown("Send any request to the Flask API and inspect exactly what comes back — including deliberately broken ones.")
    choice = st.selectbox("Preset request", list(PRESETS))
    method, path, body = PRESETS[choice]
    a, b = st.columns([1, 3])
    a.text_input("Method", method, disabled=True)
    b.text_input("Path", path, disabled=True)
    body_text = st.text_area("Request body", body, height=110, disabled=(method == "GET"), key=f"body_{choice}")
    if st.button("Send request", type="primary", disabled=False):
        r = client.custom(method, path, body_text if method != "GET" else None)
        st.session_state.explorer = r
    r = st.session_state.get("explorer")
    if r is not None:
        if r.unreachable:
            st.error(r.message)
        else:
            good = r.status < 300
            st.markdown(f"**HTTP {r.status}** · {r.latency_ms:.1f} ms · "
                        + ("✅ success" if good else "⚠️ handled error (structured JSON, server stays up)"))
            st.json(r.data)

# ======================== Request log ========================
with tab_log:
    log = st.session_state.api_log
    if not log:
        st.info("No requests yet. Make a prediction and every call to the Flask API appears here.")
    else:
        df = pd.DataFrame(log)
        ok = df["status"].apply(lambda s: isinstance(s, int) and s < 300)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Requests", len(df)); m2.metric("Successful", int(ok.sum()))
        m3.metric("Mean latency", f"{df['latency_ms'].mean():.1f} ms"); m4.metric("Slowest", f"{df['latency_ms'].max():.1f} ms")
        st.dataframe(df.iloc[::-1], width="stretch", hide_index=True, height=480)
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Download log (CSV)", df.to_csv(index=False), "api_request_log.csv", "text/csv", width="stretch")
        if c2.button("Clear log", width="stretch"):
            st.session_state.api_log = []
            st.rerun()

st.markdown('<div class="footer">Task 8 · Streamlit frontend integrated with the Flask API from Task 4</div>', unsafe_allow_html=True)
