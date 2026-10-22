"""
Catalyst QuickML LLM Serving (Zoho-hosted models such as GLM-4.7-Flash).

The endpoint needs a Zoho OAuth access token with the QuickML.deployment.READ scope.
Access tokens last an hour, so a long-lived refresh token from a Zoho "Self Client"
is kept in the environment and swapped for a fresh access token when needed.
"""
import re
import threading
import time

import requests

_lock = threading.Lock()
_token = {"value": "", "expires": 0.0}


class QuickMLError(Exception):
    pass


def configured(cfg):
    return bool(cfg and cfg.get("url") and cfg.get("endpoint_key") and cfg.get("org")
                and cfg.get("client_id") and cfg.get("client_secret") and cfg.get("refresh_token"))


def access_token(cfg):
    with _lock:
        if _token["value"] and time.time() < _token["expires"] - 60:
            return _token["value"]
        resp = requests.post(
            cfg["accounts"].rstrip("/") + "/oauth/v2/token",
            params={"refresh_token": cfg["refresh_token"], "client_id": cfg["client_id"],
                    "client_secret": cfg["client_secret"], "grant_type": "refresh_token"},
            timeout=15)
        data = resp.json() if resp.status_code == 200 else {}
        if not data.get("access_token"):
            raise QuickMLError(f"Zoho token refresh failed ({resp.status_code}: {str(data.get('error', ''))[:60]})")
        _token["value"] = data["access_token"]
        _token["expires"] = time.time() + float(data.get("expires_in", 3600))
        return _token["value"]


def first_text(node):
    """The reply sits at data[0].data; dig for the first string so small format changes don't break it."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        for item in node:
            text = first_text(item)
            if text:
                return text
    if isinstance(node, dict):
        for key in ("data", "response", "output", "text", "content"):
            if key in node:
                text = first_text(node[key])
                if text:
                    return text
    return ""


def strip_thinking(text):
    """Reasoning models may wrap their thinking in <think> tags before the answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def generate(cfg, prompt, timeout=25):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Zoho-oauthtoken {access_token(cfg)}",
        "Environment": cfg.get("environment") or "Development",
        "x-quickml-endpoint-key": cfg["endpoint_key"],
        "CATALYST-ORG": cfg["org"],
    }
    resp = requests.post(cfg["url"], headers=headers, json={"prompt": prompt}, timeout=timeout)
    if resp.status_code == 401:
        _token["value"] = ""          # token revoked or expired early; refresh next time
    if resp.status_code != 200:
        raise QuickMLError(f"QuickML call failed ({resp.status_code})")
    text = strip_thinking(first_text(resp.json()))
    if not text:
        raise QuickMLError("QuickML returned an empty reply")
    return text
