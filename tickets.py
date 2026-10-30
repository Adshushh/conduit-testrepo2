"""
Open tickets as searchable text, so Conduit can spot a bug that was already reported.

Each project gets its own Pinecone namespace, "tickets-<project namespace>", in the
same index as the code. One record per open ticket: the id is the Cliq record id,
the text is the title plus description. Closed and won't-fix tickets are removed.
"""
import requests

import index_repo

DEFAULT_THRESHOLD = 0.87   # multilingual-e5-large scores sit high; real duplicates land above this
MAX_TEXT = 1500


def namespace_for(project_namespace):
    return f"tickets-{project_namespace}"[:64]


def add(host, api_key, project_namespace, record_id, ticket_id, title, text):
    body = f"{title}. {text}".strip()[:MAX_TEXT]
    record = {
        "_id": str(record_id),
        index_repo.TEXT_FIELD: body,
        "ticket_id": str(ticket_id),
        "title": str(title)[:200],
    }
    index_repo.upsert(host, api_key, namespace_for(project_namespace), [record])


def remove(host, api_key, project_namespace, record_id):
    resp = requests.post(
        f"https://{host}/vectors/delete",
        headers=index_repo.pinecone_headers(api_key),
        json={"ids": [str(record_id)], "namespace": namespace_for(project_namespace)},
        timeout=30,
    )
    return resp.status_code in (200, 202, 404)


def similar(host, api_key, project_namespace, title, text, threshold=DEFAULT_THRESHOLD, exclude=None, info=None):
    """Open tickets that look like this bug. `info`, if given, receives the best score seen."""
    query = f"{title}. {text}".strip()[:MAX_TEXT]
    body = {
        "query": {"inputs": {"text": query}, "top_k": 5},
        "fields": ["ticket_id", "title"],
    }
    resp = requests.post(
        f"https://{host}/records/namespaces/{namespace_for(project_namespace)}/search",
        headers=index_repo.pinecone_headers(api_key), json=body, timeout=60,
    )
    if resp.status_code == 404:
        return []           # no tickets stored for this project yet
    if resp.status_code != 200:
        raise index_repo.IndexerError(f"Pinecone search failed {resp.status_code}: {resp.text[:200]}")
    matches = []
    for hit in resp.json().get("result", {}).get("hits", []):
        record_id = str(hit.get("_id", ""))
        score = float(hit.get("_score", 0) or 0)
        if info is not None and record_id != str(exclude or ""):
            info["best"] = max(info.get("best", 0), round(score, 3))
        if not record_id or record_id == str(exclude or "") or score < threshold:
            continue
        fields = hit.get("fields", {}) or {}
        matches.append({
            "record_id": record_id,
            "ticket_id": fields.get("ticket_id", ""),
            "title": fields.get("title", ""),
            "score": round(score, 3),
        })
    return matches[:3]


def forget(host, api_key, project_namespace):
    return index_repo.clear_namespace(host, api_key, namespace_for(project_namespace))
