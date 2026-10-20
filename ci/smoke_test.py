"""Pre-deploy check: the built indexer imports and its basic routes answer.

Run from the build directory, so it tests exactly what gets deployed:
    cd build && python ci/smoke_test.py
"""
import os
import sys

os.environ.setdefault("CONDUIT_SECRET", "ci-smoke-test")
os.environ.setdefault("PINECONE_API_KEY", "ci-smoke-test")
os.environ["TICK_MINUTES"] = "0"
sys.path.insert(0, os.getcwd())

import app  # noqa: E402

client = app.app.test_client()
failures = []

health = client.get("/")
if health.status_code != 200 or health.get_json().get("ok") is not True:
    failures.append(f"GET / -> {health.status_code} {health.get_data(as_text=True)[:200]}")
for path in ("/privacy", "/terms", "/flux.png"):
    resp = client.get(path)
    if resp.status_code != 200:
        failures.append(f"GET {path} -> {resp.status_code}")

if failures:
    print("Smoke test failed:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("Smoke test passed.")
