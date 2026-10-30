"""
Open tickets as searchable text, so Conduit can spot a bug that was already reported.

Each project gets its own Pinecone namespace, "tickets-<project namespace>", in the
same index as the code. One record per open ticket: the id is the Cliq record id,
the text is the title plus description. Closed and won't-fix tickets are removed.
"""
import requests

import index_repo









def similar(host, api_key, project_namespace, title, text, threshold=DEFAULT_THRESHOLD, exclude=None, info=None):
    """Open tickets that look like this bug. `info`, if given, receives the best score seen."""
    query = f"{title}. {text}".strip()[:MAX_TEXT]
    resp = requests.post(
        f"https://{host}/records/namespaces/{namespace_for(project_namespace)}/search",
        headers=index_repo.pinecone_headers(api_key), json=body, timeout=60,
    )
    if resp.status_code != 200:
        raise index_repo.IndexerError(f"Pinecone search failed {resp.status_code}: {resp.text[:200]}")
    for hit in resp.json().get("result", {}).get("hits", []):
        score = float(hit.get("_score", 0) or 0)
        if not record_id or record_id == str(exclude or "") or score < threshold:
            continue
        matches.append({
            "record_id": record_id,
            "ticket_id": fields.get("ticket_id", ""),
            "title": fields.get("title", ""),
            "score": round(score, 3),
        })


def forget(host, api_key, project_namespace):
    return index_repo.clear_namespace(host, api_key, namespace_for(project_namespace))
