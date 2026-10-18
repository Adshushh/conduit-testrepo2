"""
Orgs that turned on reminders.

Each org's admin gives Conduit an extension token from their own Cliq; with it the
indexer can tick that org's copy of Conduit. The tokens live in a small Pinecone
namespace, one record per tenant, encrypted with a key derived from CONDUIT_SECRET
so the stored text is useless on its own.

Encryption uses only the standard library: HMAC-SHA256 as a keystream in counter
mode, with a separate HMAC tag so tampered records are ignored.
"""
import base64
import hashlib
import hmac
import json
import os

import requests

import index_repo

NAMESPACE = "__registry"


def _keys(secret):
    enc = hmac.new(secret.encode(), b"conduit-registry-enc", hashlib.sha256).digest()
    mac = hmac.new(secret.encode(), b"conduit-registry-mac", hashlib.sha256).digest()
    return enc, mac


def _stream(key, nonce, length):
    out = b""
    counter = 0
    while len(out) < length:
        out += hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        counter += 1
    return out[:length]


def seal(secret, data):
    enc, mac = _keys(secret)
    plain = json.dumps(data).encode()
    nonce = os.urandom(16)
    cipher = bytes(a ^ b for a, b in zip(plain, _stream(enc, nonce, len(plain))))
    tag = hmac.new(mac, nonce + cipher, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(nonce + tag + cipher).decode()


def unseal(secret, blob):
    enc, mac = _keys(secret)
    try:
        raw = base64.urlsafe_b64decode(blob.encode())
    except (ValueError, TypeError):
        return None
    nonce, tag, cipher = raw[:16], raw[16:48], raw[48:]
    if len(raw) < 48 or not hmac.compare_digest(tag, hmac.new(mac, nonce + cipher, hashlib.sha256).digest()):
        return None
    plain = bytes(a ^ b for a, b in zip(cipher, _stream(enc, nonce, len(cipher))))
    try:
        return json.loads(plain)
    except ValueError:
        return None


def record_id(tenant):
    return tenant or "host"


def save(host, api_key, secret, tenant, endpoint, token):
    record = {
        "_id": record_id(tenant),
        index_repo.TEXT_FIELD: "conduit org",
        "blob": seal(secret, {"endpoint": endpoint, "token": token}),
    }
    index_repo.upsert(host, api_key, NAMESPACE, [record])


def remove(host, api_key, tenant):
    resp = requests.post(
        f"https://{host}/vectors/delete",
        headers=index_repo.pinecone_headers(api_key),
        json={"ids": [record_id(tenant)], "namespace": NAMESPACE},
        timeout=30,
    )
    return resp.status_code in (200, 202, 404)


def load(host, api_key, secret):
    """Every connected org as a list of {"tenant", "endpoint", "token"}."""
    headers = index_repo.pinecone_headers(api_key)
    ids, token = [], None
    while True:
        params = {"namespace": NAMESPACE, "limit": 100}
        if token:
            params["paginationToken"] = token
        resp = requests.get(f"https://{host}/vectors/list", headers=headers, params=params, timeout=30)
        if resp.status_code == 404:
            return []
        if resp.status_code != 200:
            raise index_repo.IndexerError(f"Pinecone list failed {resp.status_code}")
        body = resp.json()
        ids += [v["id"] for v in body.get("vectors", []) if v.get("id")]
        token = (body.get("pagination") or {}).get("next")
        if not token:
            break
    orgs = []
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        resp = requests.get(f"https://{host}/vectors/fetch", headers=headers,
                            params=[("namespace", NAMESPACE)] + [("ids", i) for i in batch], timeout=30)
        if resp.status_code != 200:
            raise index_repo.IndexerError(f"Pinecone fetch failed {resp.status_code}")
        for rid, vec in (resp.json().get("vectors") or {}).items():
            data = unseal(secret, str((vec.get("metadata") or {}).get("blob", "")))
            if data and data.get("endpoint") and data.get("token"):
                orgs.append({"tenant": "" if rid == "host" else rid,
                             "endpoint": data["endpoint"], "token": data["token"]})
    return orgs
