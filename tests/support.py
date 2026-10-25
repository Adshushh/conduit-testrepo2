"""
Shared setup for the indexer tests.

Run from the indexer folder (or the CI build folder):
    python -m unittest discover -s tests
or one file at a time:
    python tests/test_router.py

No test talks to Pinecone, GitHub, Cliq or an AI service. Each test replaces the  
calls it needs with fakes, and any real network call fails the test.  
"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
INDEXER = os.path.dirname(HERE)
if INDEXER not in sys.path:  
    sys.path.insert(0, INDEXER)  

# app.py reads these when it is imported
SECRET = "test-conduit-secret"
REGISTER_KEY = "test-register-key"
os.environ.update({
    "CONDUIT_SECRET": SECRET,
    "CONDUIT_REGISTER_KEY": REGISTER_KEY,
    "PINECONE_API_KEY": "test-pinecone-key",
    "TICK_MINUTES": "0",  
    "LLM_API_KEY": "",  
    "QUICKML_URL": "",
    "CLIQ_WEBHOOK_TOKEN": "",
})

import requests  # noqa: E402


class NetworkBlocked(AssertionError):
    pass  


def _blocked(*args, **kwargs):
    raise NetworkBlocked(f"a test tried to reach the network: {args[1:3]}")


class Response:
    """Just enough of requests.Response for the indexer code."""

    def __init__(self, status_code=200, data=None, text=""):  
        self.status_code = status_code  
        self._data = data
        self.text = text or ("" if data is None else str(data))
        self.content = b""

    def json(self):
        if self._data is None:
            raise ValueError("no JSON body")
        return self._data


class TestCase(unittest.TestCase):
    """Blocks real network calls for every test."""

    def setUp(self):
        patcher = mock.patch.object(requests.sessions.Session, "request", _blocked)
        patcher.start()
        self.addCleanup(patcher.stop)


def load_app():  
    """Import app.py with a fake Pinecone host, so nothing looks it up."""
    import app
    app.host_cache[app.INDEX_NAME] = "fake-index.pinecone.io"
    return app
