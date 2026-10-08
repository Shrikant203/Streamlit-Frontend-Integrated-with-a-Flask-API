"""
api_client.py
=============
HTTP client used by the Streamlit frontend to talk to the Flask API (Task 4).

The frontend never loads the model: every prediction is a JSON request to the
Flask backend. This module wraps those requests so that
  * every call returns an APIResult (never raises) - connection problems,
    timeouts and HTTP errors are all reported as data the UI can display;
  * the round-trip latency of every call is measured;
  * every call can be recorded in the UI's request log through a callback.
"""
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests


@dataclass
class APIResult:
    ok: bool                      # True only for a 2xx response
    status: Optional[int]         # HTTP status, or None if the server was unreachable
    data: Any                     # parsed JSON body (dict) or None
    latency_ms: float             # full round-trip time measured by the client
    method: str
    path: str
    request_body: Any = None      # what was sent (for display in the UI)
    error: Optional[str] = None   # connection-level problem (offline / timeout / ...)

    @property
    def unreachable(self) -> bool:
        return self.status is None

    @property
    def message(self) -> str:
        """Human-readable one-line summary of a failed call."""
        if self.unreachable:
            return self.error or "Cannot reach the API."
        if isinstance(self.data, dict) and "error" in self.data:
            return f"HTTP {self.status}: {self.data['error']}"
        return f"HTTP {self.status}"


class DigitAPIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 5.0,
                 on_call: Optional[Callable[[APIResult], None]] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.on_call = on_call
        self.session = requests.Session()

    # ------------------------------------------------------------------
    def _call(self, method: str, path: str, json_body: Any = None, raw_body: Optional[str] = None,
              record: bool = True) -> APIResult:
        url = self.base_url + path
        sent = json_body if raw_body is None else raw_body
        t0 = time.perf_counter()
        try:
            if raw_body is not None:        # used by the API explorer to send deliberately broken JSON
                resp = self.session.request(method, url, data=raw_body.encode("utf-8"),
                                            headers={"Content-Type": "application/json"}, timeout=self.timeout)
            else:
                resp = self.session.request(method, url, json=json_body, timeout=self.timeout)
            ms = (time.perf_counter() - t0) * 1000
            try:
                data = resp.json()
            except ValueError:
                data = {"error": "Server returned a non-JSON response.", "text": resp.text[:200]}
            result = APIResult(resp.ok, resp.status_code, data, ms, method, path, sent)
        except requests.exceptions.ConnectionError:
            result = APIResult(False, None, None, (time.perf_counter() - t0) * 1000, method, path, sent,
                               error=f"Cannot reach the Flask API at {self.base_url} (connection refused).")
        except requests.exceptions.Timeout:
            result = APIResult(False, None, None, (time.perf_counter() - t0) * 1000, method, path, sent,
                               error=f"The Flask API did not answer within {self.timeout:g} s (timeout).")
        except requests.exceptions.RequestException as e:
            result = APIResult(False, None, None, (time.perf_counter() - t0) * 1000, method, path, sent,
                               error=f"Request failed: {e}")
        if record and self.on_call:
            self.on_call(result)
        return result

    # ------------------------------------------------------------------ endpoints
    def info(self) -> APIResult:
        return self._call("GET", "/")

    def health(self, record: bool = False) -> APIResult:
        return self._call("GET", "/health", record=record)

    @staticmethod
    def _payload(pixels) -> list:
        """8x8 array -> flat list of 64 numbers in the 0-16 range (the API's native scale)."""
        flat = [min(16.0, max(0.0, round(float(v), 4))) for v in list(_flatten(pixels))]
        return flat

    def predict(self, pixels) -> APIResult:
        return self._call("POST", "/predict", {"image": self._payload(pixels)})

    def predict_batch(self, images) -> APIResult:
        return self._call("POST", "/predict/batch", {"images": [self._payload(p) for p in images]})

    def custom(self, method: str, path: str, body_text: Optional[str] = None) -> APIResult:
        """Free-form request used by the API explorer (body is sent exactly as typed)."""
        if method == "GET" or not body_text:
            return self._call(method, path)
        return self._call(method, path, raw_body=body_text)


def _flatten(x):
    try:
        import numpy as np
        return np.asarray(x, dtype=float).reshape(-1).tolist()
    except Exception:
        return [v for row in x for v in row]
