"""
Conduit routing: decide which code a bug is about and who should own it.

    1. Search Pinecone for the chunks closest to the bug text.
    2. Optionally ask an LLM which chunks are really responsible, for a short
       triage note, and for questions to ask the reporter when the report is thin.
    3. Run git blame on the top 3 files (only the matched lines) and score each
       author: lines they own x file weight (3, 2, 1). Each line counts by the age
       of its commit: full when new, half every 90 days, at least 0.2.
    4. Map authors to Cliq users through the "people" table Conduit keeps.
       Owners named in the repo's CODEOWNERS file for a picked file get a bonus,
       weighted like the file, so they count even when blame is a close call.
    5. Rate the confidence of each candidate, in two parts:
       where: how sure the AI is that the cause is in the files it picked
              (no AI: a middle value based on code search only)
       who:   the candidate's share of the blamed code (capped when there is little code)
       Each candidate's confidence is the lower of the two. The extension compares
       it with the org's minimum and sends low-confidence bugs to the team lead.
"""
import fnmatch
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

import index_repo
import quickml


# the same, on a chosen branch
# GitHub's private commit emails carry the username: 12345+login@users.noreply.github.com
# keep the whole /route call inside the time Cliq waits for an answer
# the whole /route call must answer before Cliq stops waiting for the form, so every
# step only gets the time left; the AI is skipped rather than blame, which routing needs
# "where" confidence for each level the AI can give
# CODEOWNERS: where GitHub looks for it, the bonus per owner (in lines, times the file weight)
# fewer blamed lines than this is too little to be sure who owns the code

# Old code counts less: a blamed line loses half its weight every 90 days, down to 0.2.










# ---------------------------------------------------------------- search



# ---------------------------------------------------------------- LLM

def llm_triage(llm, title, description, severity, hits, ask_questions, deadline=None):
    """Return {"ranked": [hit indexes], "note": str, "questions": [str],
    "code_level": "high"|"medium"|"low"|"", "code_reason": str}, {"error": str}, or None."""


    # Zoho's own model first (Catalyst QuickML), then the OpenAI-compatible fallback (Groq)
    if not isinstance(data, dict):
        return {"error": "; ".join(problems) or "AI reply was not JSON"}
    ranked = []
    for item in data.get("ranked", []) or []:
        index = to_index(item, hits)
        if index is not None:
            ranked.append(index)
    questions = [str(q).strip()[:100] for q in data.get("questions", []) if str(q).strip()][:3]
    level = str(data.get("confidence", "")).strip().lower()
    return {"ranked": list(dict.fromkeys(ranked))[:3],
            "note": clean_note(str(data.get("note", "")), hits)[:400],
            "questions": questions if ask_questions else [],
            "model": model,
            "code_level": level if level in CODE_CONFIDENCE else "",
            "code_reason": clean_note(str(data.get("confidence_reason", "")), hits)[:120]}






def chat_body(llm, prompt):
    body = {"model": llm["model"], "temperature": 0,
            "messages": [{"role": "system",
                          "content": "You triage bug reports for a software team. Reply with JSON only."},
                         {"role": "user", "content": prompt}]}
    # gpt-oss models think before answering; keep that short so Cliq doesn't time out
    if "gpt-oss" in llm["model"]:
        body["reasoning_effort"] = "low"
    return body




# ---------------------------------------------------------------- blame





def blame_file(owner, repo, token, path, start, end, ref=None, timeout=12, now=None):
    """Return {author_key: {"lines": n, "weighted": w, "name", "email", "login"}} for lines
    start..end, where weighted counts each line by the age of its commit.
    Uses the repository's default branch, or `ref` when a branch was chosen."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(GITHUB_GRAPHQL, headers=headers, timeout=timeout, json={
        "query": query, "variables": variables})
    if resp.status_code == 401:
        raise RouteError("GitHub blame failed (401): the project token was rejected")
    data = payload.get("data") or {}
    try:
        ranges = repository[field]["target"]["blame"]["ranges"]
    except (KeyError, TypeError):
        reason = ""
        for err in payload.get("errors") or []:
            reason = str(err.get("message") or err.get("type") or "")[:160]
            break
        raise RouteError(f"GitHub returned no blame for {path}" + (f": {reason}" if reason else ""))
    return owners


# ---------------------------------------------------------------- CODEOWNERS



def codeowners_match(pattern, path):
    """GitHub CODEOWNERS matching (gitignore style, simplified)."""
    anchored = pattern.startswith("/")
    body = pattern.strip("/")
    if not body:
        return False
    is_dir = pattern.endswith("/")
    if "/" not in body and not anchored:
        # a bare name or glob matches a file or folder name at any depth
        segments = path.split("/")
        if is_dir:
            segments = segments[:-1]
        return any(fnmatch.fnmatchcase(seg, body) for seg in segments)
    regex = ""
    i = 0
    while i < len(body):
        if body.startswith("**", i):
            regex += ".*"
            i += 2
        elif body[i] == "*":
            regex += "[^/]*"
            i += 1
        elif body[i] == "?":
            regex += "[^/]"
            i += 1
        else:
            regex += re.escape(body[i])
            i += 1
    return re.fullmatch(regex + "(/.*)?", path) is not None






# ---------------------------------------------------------------- people



# ---------------------------------------------------------------- confidence



def rate_candidates(candidates, code_percent, blamed_lines):
    """Who: each candidate's share of the blamed code, then the lower of where and who."""
    total = sum(c["score"] for c in candidates)
    thin = blamed_lines < THIN_LINES
    for cand in candidates:
        share = round(100 * cand["score"] / total) if total else 0
        if thin:
            share = min(share, THIN_CAP)
        cand["share"] = share
        cand["confidence"] = min(share, code_percent)
    return thin


# ---------------------------------------------------------------- main



def route(req, host, pinecone_key, llm=None):
    warnings = []

    if not hits:
        raise RouteError(f"no matching code; has namespace {req['namespace']} been indexed?")

    triaged = clock()
    blame_seconds = max(MIN_STEP_SECONDS, min(12, time_left(deadline) - 1))

    branch_missing = []


    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(blame_one, enumerate(files)))
    blamed = clock()
    if branch_missing:
        warnings.append(f"{branch_missing[0]}; used the default branch for blame")

    linked_emails = set(people.values())

    # CODEOWNERS: a bonus for the declared owners of each picked file
    owners_source, owner_rules = fetch_codeowners(repo_full, token, ref or None,
                                                  max(2, min(6, time_left(deadline))))
    owners_by_file = {}
    for index, hit in enumerate(files):
        declared = owners_for(owner_rules, hit["path"]) if owner_rules else []
        if declared:
            owners_by_file[hit["path"]] = declared
        for owner in declared:
            info = {"login": "" if "@" in owner else owner, "email": owner if "@" in owner else ""}
            cliq = match_cliq(info, people, linked_emails)
            merge_key = cliq or owner
            cand = scores.setdefault(merge_key, {"key": owner, "name": owner, "login": info["login"],
                                                 "email": info["email"], "cliq": cliq, "score": 0,
                                                 "files": []})
            cand["score"] += CODEOWNER_BONUS * FILE_WEIGHTS[index]
            cand["codeowner"] = True
            if hit["path"] not in cand["files"]:
                cand["files"].append(hit["path"])

