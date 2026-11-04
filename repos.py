"""
Repositories Conduit keeps up to date by itself.

When a repo is indexed, the indexer remembers it here: the GitHub repo, the branch,
the token and the webhook it created. GitHub then calls /github/<key> on every push,
and only the files that changed are indexed again.

Each project has two code namespaces, its "slots": <namespace> and <namespace>--b.
/route always searches the active one. A full re-index builds the other slot and
switches to it when it is done, so bug reports never meet an empty index.

Records live in the Pinecone namespace "__repos", one per project, sealed with a key
derived from CONDUIT_SECRET (see registry.py), so the stored tokens are useless on
their own.
"""
import hashlib
import hmac
import secrets
import threading
import time

import requests

import index_repo
import registry

NAMESPACE = "__repos"
CACHE_SECONDS = 60
# GitHub sends at most 20 commits in a push event; more means the file lists are incomplete
MAX_PUSH_COMMITS = 20
# a push touching more files than this is cheaper to rebuild in full
MAX_CHANGED_FILES = 300

_cache = {}
_cache_lock = threading.Lock()


def repo_key(tenant, namespace):
    """Stable, opaque id for one project of one org; also the webhook URL path."""
    return hashlib.sha256(f"{tenant}:{namespace}".encode()).hexdigest()[:24]


def physical(scoped_namespace, slot):
    return scoped_namespace if slot != "b" else f"{scoped_namespace}--b"


def other_slot(slot):
    return "a" if slot == "b" else "b"


def load(host, api_key, secret, key):
    """The stored record for a project, or None."""
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
            return dict(hit[1]) if hit[1] else None
    resp = requests.get(f"https://{host}/vectors/fetch", headers=index_repo.pinecone_headers(api_key),
                        params=[("namespace", NAMESPACE), ("ids", key)], timeout=30)
    data = None
    if resp.status_code == 200:
        vec = (resp.json().get("vectors") or {}).get(key)
        if vec:
            data = registry.unseal(secret, str((vec.get("metadata") or {}).get("blob", "")))
    elif resp.status_code != 404:
        raise index_repo.IndexerError(f"Pinecone fetch failed {resp.status_code}")
    with _cache_lock:
        _cache[key] = (time.monotonic(), data)
    return dict(data) if data else None


def save(host, api_key, secret, key, data):
    record = {"_id": key, index_repo.TEXT_FIELD: "conduit repo", "blob": registry.seal(secret, data)}
    index_repo.upsert(host, api_key, NAMESPACE, [record])
    with _cache_lock:
        _cache[key] = (time.monotonic(), dict(data))


def remove(host, api_key, key):
    requests.post(f"https://{host}/vectors/delete", headers=index_repo.pinecone_headers(api_key),
                  json={"ids": [key], "namespace": NAMESPACE}, timeout=30)
    with _cache_lock:
        _cache.pop(key, None)


def active_namespace(host, api_key, secret, scoped_namespace, tenant, namespace):
    """Where /route should search: the active slot, or the plain namespace for older projects."""
    try:
        data = load(host, api_key, secret, repo_key(tenant, namespace))
    except (index_repo.IndexerError, requests.RequestException):
        data = None
    return physical(scoped_namespace, (data or {}).get("slot", "a"))


# ---------------------------------------------------------------- GitHub webhooks

def _github_headers(token):
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def create_hook(repo, token, url, hook_secret):
    """(hook id, "") when GitHub accepted the webhook, else (None, reason for the admin)."""
    body = {"name": "web", "active": True, "events": ["push"],
            "config": {"url": url, "content_type": "json", "secret": hook_secret, "insecure_ssl": "0"}}
    try:
        resp = requests.post(f"https://api.github.com/repos/{repo}/hooks", headers=_github_headers(token),
                             json=body, timeout=20)
    except requests.RequestException:
        return None, "GitHub could not be reached"
    if resp.status_code == 201:
        return resp.json().get("id"), ""
    if resp.status_code in (403, 404):
        return None, "the token needs the Webhooks: Read and write permission"
    if resp.status_code == 422:
        return None, "GitHub refused the webhook (it may already exist or the repo has too many)"
    return None, f"GitHub answered {resp.status_code}"


def delete_hook(repo, token, hook_id):
    if not hook_id:
        return
    try:
        requests.delete(f"https://api.github.com/repos/{repo}/hooks/{hook_id}",
                        headers=_github_headers(token), timeout=20)
    except requests.RequestException:
        pass


def new_hook_secret():
    return secrets.token_hex(20)


def signature_ok(hook_secret, body, header):
    """GitHub signs each delivery: X-Hub-Signature-256: sha256=<hmac of the raw body>."""
    if not hook_secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(hook_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[7:])


def read_push(payload, tracked_ref):
    """What a push means for the index.

    Returns None when it can be ignored (another branch, a deleted branch), else
    {"full": bool, "changed": set, "removed": set, "head": sha}.
    """
    branch = tracked_ref or (payload.get("repository") or {}).get("default_branch") or ""
    if payload.get("ref") != f"refs/heads/{branch}" or payload.get("deleted"):
        return None
    commits = payload.get("commits") or []
    changed, removed = set(), set()
    for commit in commits:
        for path in commit.get("added", []) + commit.get("modified", []):
            changed.add(path)
            removed.discard(path)
        for path in commit.get("removed", []):
            removed.add(path)
            changed.discard(path)
    full = bool(payload.get("forced")) or len(commits) >= MAX_PUSH_COMMITS \
        or len(changed) + len(removed) > MAX_CHANGED_FILES
    return {"full": full, "changed": changed, "removed": removed,
            "head": (payload.get("after") or "")[:12]}
