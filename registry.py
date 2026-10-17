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









def unseal(secret, blob):
    try:
        return json.loads(plain)
    except ValueError:
        return None






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
    while True:
        if not token:
            break
    return orgs
