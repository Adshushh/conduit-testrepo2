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

GITHUB_GRAPHQL = "https://api.github.com/graphql"
FILE_WEIGHTS = (3, 2, 1)
MAX_CANDIDATES = 5
SNIPPET_CHARS = 900

BLAME_FIELDS = ("{target{... on Commit{blame(path:$path){ranges{startingLine endingLine "
                "commit{authoredDate author{name email user{login}}}}}}}}")
BLAME_QUERY = (
    "query($owner:String!,$name:String!,$path:String!){repository(owner:$owner,name:$name)"
    "{defaultBranchRef" + BLAME_FIELDS + "}}"
)
# the same, on a chosen branch
BLAME_QUERY_REF = (
    "query($owner:String!,$name:String!,$path:String!,$ref:String!){repository(owner:$owner,name:$name)"
    "{ref(qualifiedName:$ref)" + BLAME_FIELDS + "}}"
)
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
# GitHub's private commit emails carry the username: 12345+login@users.noreply.github.com
NOREPLY_RE = re.compile(r"^(?:\d+\+)?([a-z0-9-]+)@users\.noreply\.github\.com$")
# keep the whole /route call inside the time Cliq waits for an answer
QUICKML_TIMEOUT = 12
LLM_TIMEOUT = 10
# the whole /route call must answer before Cliq stops waiting for the form, so every
# step only gets the time left; the AI is skipped rather than blame, which routing needs
ROUTE_BUDGET = float(os.environ.get("ROUTE_BUDGET", "20") or 20)
BLAME_RESERVE = 6
MIN_AI_SECONDS = 4
MIN_STEP_SECONDS = 3
# "where" confidence for each level the AI can give
CODE_CONFIDENCE = {"high": 90, "medium": 65, "low": 35}
NO_AI_CONFIDENCE = 65
# CODEOWNERS: where GitHub looks for it, the bonus per owner (in lines, times the file weight)
CODEOWNERS_PATHS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")
CODEOWNER_BONUS = 10
CODEOWNERS_TTL = 600
_codeowners_cache = {}
_codeowners_lock = threading.Lock()
# fewer blamed lines than this is too little to be sure who owns the code
THIN_LINES = 5
THIN_CAP = 50

# Old code counts less: a blamed line loses half its weight every 90 days, down to 0.2.
RECENCY_HALF_LIFE_DAYS = 90
RECENCY_FLOOR = 0.2


def clock():
    return time.monotonic()


def time_left(deadline):
    return float("inf") if deadline is None else deadline - clock()


class RouteError(Exception):
    pass


class BranchMissing(RouteError):
    """The chosen branch doesn't exist (renamed or deleted); blame falls back to the default branch."""


# ---------------------------------------------------------------- search

def search_chunks(host, api_key, namespace, text, top_k=8, timeout=15):
    body = {
        "query": {"inputs": {"text": text}, "top_k": top_k},
        "fields": ["path", "lines", index_repo.TEXT_FIELD],
    }
    resp = requests.post(f"https://{host}/records/namespaces/{namespace}/search",
                         headers=index_repo.pinecone_headers(api_key), json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RouteError(f"code search failed (Pinecone said {resp.status_code})")
    hits = []
    for hit in resp.json().get("result", {}).get("hits", []):
        fields = hit.get("fields", {})
        if fields.get("path") and fields.get("lines"):
            hits.append({"path": fields["path"], "lines": str(fields["lines"]),
                         "text": fields.get(index_repo.TEXT_FIELD, ""),
                         "score": hit.get("_score", 0)})
    return hits


# ---------------------------------------------------------------- LLM

def llm_triage(llm, title, description, severity, hits, ask_questions, deadline=None):
    """Return {"ranked": [hit indexes], "note": str, "questions": [str],
    "code_level": "high"|"medium"|"low"|"", "code_reason": str}, {"error": str}, or None."""
    if not (llm and (llm.get("key") or quickml.configured(llm.get("quickml"))) and hits):
        return None
    snippets = []
    for i, hit in enumerate(hits):
        code = hit["text"]
        if code.startswith("File: "):
            code = code.split("\n", 1)[1] if "\n" in code else ""
        snippets.append(f"[{i}] {hit['path']} (lines {hit['lines']})\n{code[:SNIPPET_CHARS]}")
    ask = (
        "If the report lacks something a developer needs to find the bug (steps to reproduce, "
        "expected vs actual behaviour, which page or screen), add up to 3 short questions for the "
        "reporter, each under 90 characters. If the report is clear enough, return an empty list."
        if ask_questions else "Return an empty list for questions."
    )
    prompt = (
        f"Bug report\nTitle: {title}\nSeverity: {severity}\nDescription: {description}\n\n"
        f"Code snippets found by search:\n\n" + "\n\n".join(snippets) + "\n\n"
        "Which snippets most likely contain the cause of this bug? Rank up to 3 snippet numbers "
        "(the number in square brackets, as an integer), best first. Write a note of at most two sentences naming the likely cause; in the note refer to files by "
        "their file name (for example favorite-button.component.ts), never by snippet number. "
        f"{ask} "
        "Also say how sure you are that the cause is in the snippets you ranked: \"high\" when the "
        "report clearly points to that code, \"medium\" when it is the best guess, \"low\" when the "
        "report is vague or the cause could be somewhere else. Add a reason under 100 characters "
        "that a team lead can read, naming files by file name.\n\n"
        "Answer with JSON only, exactly this shape:\n"
        '{"ranked": [0, 2], "note": "...", "questions": [], "confidence": "medium", "confidence_reason": "..."}'
    )
    data, model, problems = None, "", []

    def ai_seconds(limit):
        """Seconds the next AI call may take, keeping enough back for blame; 0 when too few."""
        left = time_left(deadline) - BLAME_RESERVE
        return min(limit, left) if left >= MIN_AI_SECONDS else 0

    if not ai_seconds(QUICKML_TIMEOUT):
        return {"error": "AI skipped to answer in time"}
    # Zoho's own model first (Catalyst QuickML), then the OpenAI-compatible fallback (Groq)
    if quickml.configured(llm.get("quickml")):
        try:
            data = parse_json(quickml.generate(llm["quickml"], prompt, timeout=ai_seconds(QUICKML_TIMEOUT)))
            model = llm["quickml"].get("label") or "QuickML"
        except (quickml.QuickMLError, requests.RequestException, ValueError, TypeError) as err:
            problems.append(str(err) if isinstance(err, quickml.QuickMLError) else "QuickML call failed")
            data = None
    if not isinstance(data, dict) and llm.get("key") and not ai_seconds(LLM_TIMEOUT):
        problems.append("no time left for the fallback AI")
    elif not isinstance(data, dict) and llm.get("key"):
        try:
            resp = requests.post(
                llm["base"].rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {llm['key']}", "Content-Type": "application/json",
                         "User-Agent": "conduit-indexer/1.0"},
                json=chat_body(llm, prompt),
                timeout=ai_seconds(LLM_TIMEOUT))
            if resp.status_code != 200:
                problems.append(f"AI call failed ({resp.status_code})")
            else:
                data = parse_json(resp.json()["choices"][0]["message"]["content"])
                model = llm["model"]
        except (requests.RequestException, KeyError, IndexError, ValueError, TypeError):
            problems.append("AI call failed")
            data = None
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


def clean_note(note, hits):
    """Replace leftover snippet references like ([1]) or [1] with the file name."""
    def name(match):
        i = int(match.group(1))
        return hits[i]["path"].rsplit("/", 1)[-1] if 0 <= i < len(hits) else ""
    note = re.sub(r"\(\s*\[(\d+)\]\s*\)", lambda m: "(" + name(m) + ")" if name(m) else "", note)
    note = re.sub(r"\[(\d+)\]", name, note)
    note = re.sub(r"\(?\s*snippets?\s*#?\d+(\s*(-|–|,|and|&)\s*#?\d+)*\s*\)?", " ", note, flags=re.I)
    note = re.sub(r"\(\s*\)", "", note)
    return re.sub(r"\s{2,}", " ", note).replace(" ,", ",").replace(" .", ".").strip()


def to_index(item, hits):
    """Models sometimes answer with a file name or "[2]" instead of 2; accept all of them."""
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        return item if 0 <= item < len(hits) else None
    text = str(item).strip().strip("[]").strip()
    if text.isdigit():
        number = int(text)
        return number if number < len(hits) else None
    for i, hit in enumerate(hits):
        if text and (hit["path"] == text or hit["path"].endswith("/" + text)):
            return i
    return None


def chat_body(llm, prompt):
    body = {"model": llm["model"], "temperature": 0,
            "messages": [{"role": "system",
                          "content": "You triage bug reports for a software team. Reply with JSON only."},
                         {"role": "user", "content": prompt}]}
    # gpt-oss models think before answering; keep that short so Cliq doesn't time out
    if "gpt-oss" in llm["model"]:
        body["reasoning_effort"] = "low"
    return body


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


# ---------------------------------------------------------------- blame

def is_bot(name, email, login):
    joined = f"{name} {email} {login}".lower()
    return "[bot]" in joined or "dependabot" in joined or "github-actions" in joined


def age_factor(date_text, now=None):
    """How much a blamed line counts by the age of its commit: 1 when new, half every
    RECENCY_HALF_LIFE_DAYS, never below RECENCY_FLOOR. No or bad date counts fully."""
    if not date_text:
        return 1.0
    try:
        when = datetime.fromisoformat(str(date_text).replace("Z", "+00:00"))
    except ValueError:
        return 1.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    days = max(0.0, ((now or datetime.now(timezone.utc)) - when).total_seconds() / 86400)
    return max(RECENCY_FLOOR, 0.5 ** (days / RECENCY_HALF_LIFE_DAYS))


def blame_file(owner, repo, token, path, start, end, ref=None, timeout=12, now=None):
    """Return {author_key: {"lines": n, "weighted": w, "name", "email", "login"}} for lines
    start..end, where weighted counts each line by the age of its commit.
    Uses the repository's default branch, or `ref` when a branch was chosen."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    variables = {"owner": owner, "name": repo, "path": path}
    query, field = BLAME_QUERY, "defaultBranchRef"
    if ref:
        query, field = BLAME_QUERY_REF, "ref"
        variables["ref"] = ref if ref.startswith("refs/") else f"refs/heads/{ref}"
    resp = requests.post(GITHUB_GRAPHQL, headers=headers, timeout=timeout, json={
        "query": query, "variables": variables})
    if resp.status_code == 401:
        raise RouteError("GitHub blame failed (401): the project token was rejected")
    if resp.status_code != 200:
        raise RouteError(f"GitHub blame failed ({resp.status_code})")
    payload = resp.json() or {}
    data = payload.get("data") or {}
    repository = data.get("repository")
    if ref and repository is not None and repository.get("ref") is None:
        raise BranchMissing(f"branch {ref} not found")
    try:
        ranges = repository[field]["target"]["blame"]["ranges"]
    except (KeyError, TypeError):
        reason = ""
        for err in payload.get("errors") or []:
            reason = str(err.get("message") or err.get("type") or "")[:160]
            break
        raise RouteError(f"GitHub returned no blame for {path}" + (f": {reason}" if reason else ""))
    owners = {}
    for r in ranges:
        lo, hi = max(r["startingLine"], start), min(r["endingLine"], end)
        if lo > hi:
            continue
        commit = r.get("commit") or {}
        author = commit.get("author") or {}
        name = author.get("name") or ""
        email = (author.get("email") or "").lower()
        login = ((author.get("user") or {}).get("login") or "").lower()
        if is_bot(name, email, login):
            continue
        key = login or email
        if not key:
            continue
        entry = owners.setdefault(key, {"lines": 0, "weighted": 0.0, "name": name or key,
                                        "email": email, "login": login})
        entry["lines"] += hi - lo + 1
        entry["weighted"] += (hi - lo + 1) * age_factor(commit.get("authoredDate"), now)
    return owners


# ---------------------------------------------------------------- CODEOWNERS

def parse_codeowners(text):
    """[(pattern, [owners])] in file order. Team owners (@org/team) are skipped."""
    rules = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        owners = []
        for owner in parts[1:]:
            owner = owner.strip().lower()
            if owner.startswith("@") and "/" not in owner:
                owners.append(owner[1:])
            elif "@" in owner and not owner.startswith("@"):
                owners.append(owner)
        rules.append((parts[0], owners))
    return rules


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


def owners_for(rules, path):
    """The owners of a path: the last matching rule wins, as on GitHub."""
    found = []
    for pattern, owners in rules:
        if codeowners_match(pattern, path):
            found = owners
    return found


def fetch_codeowners(repo_full, token, ref=None, timeout=6):
    """(source path, rules) from the repo, cached for a while; ("", []) when there is none."""
    cache_key = (repo_full, ref or "")
    with _codeowners_lock:
        hit = _codeowners_cache.get(cache_key)
        if hit and time.monotonic() - hit[0] < CODEOWNERS_TTL:
            return hit[1], hit[2]
    headers = {"Accept": "application/vnd.github.raw+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    source, rules = "", []
    for path in CODEOWNERS_PATHS:
        try:
            resp = requests.get(f"https://api.github.com/repos/{repo_full}/contents/{path}",
                                headers=headers, params={"ref": ref} if ref else None, timeout=timeout)
        except requests.RequestException:
            return "", []          # try again next time instead of caching a network failure
        if resp.status_code == 200:
            source, rules = path, parse_codeowners(resp.text)
            break
        if resp.status_code not in (404,):
            return "", []
    with _codeowners_lock:
        _codeowners_cache[cache_key] = (time.monotonic(), source, rules)
    return source, rules


# ---------------------------------------------------------------- people

def match_cliq(info, people, linked_emails):
    """Find the Cliq user for a blame author: by GitHub login, by the login inside a
    GitHub noreply email, by commit email, or when the commit email is itself a linked Cliq email."""
    login, email = info.get("login") or "", info.get("email") or ""
    if login and people.get(login):
        return people[login]
    noreply = NOREPLY_RE.match(email)
    if noreply and people.get(noreply.group(1)):
        return people[noreply.group(1)]
    if email and people.get(email):
        return people[email]
    if email and email in linked_emails:
        return email
    return ""


# ---------------------------------------------------------------- confidence

def code_confidence(ai):
    """Where: how sure we are that the cause is in the files that were blamed."""
    if ai and "error" not in ai and ai.get("code_level"):
        level = ai["code_level"]
        return {"level": level, "percent": CODE_CONFIDENCE[level],
                "reason": ai.get("code_reason", ""), "source": "ai"}
    reason = "AI unavailable; based on code search only"
    if ai and "error" not in ai:
        reason = "AI gave no confidence; based on code search only"
    return {"level": "medium", "percent": NO_AI_CONFIDENCE, "reason": reason, "source": "search"}


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

def pick_files(hits, ranked):
    """Up to 3 distinct files, in the order the AI (or the search) ranked them."""
    order = list(ranked) + [i for i in range(len(hits)) if i not in ranked]
    files, seen = [], set()
    for i in order:
        hit = hits[i]
        if hit["path"] in seen:
            continue
        seen.add(hit["path"])
        files.append(hit)
        if len(files) == 3:
            break
    return files


def route(req, host, pinecone_key, llm=None):
    repo_full = req["repo"]
    owner, repo = repo_full.split("/", 1)
    token = req.get("token", "")
    ref = str(req.get("ref") or "").strip()
    if ref and not BRANCH_RE.match(ref):
        ref = ""
    people = {str(k).lower(): str(v).lower() for k, v in (req.get("people") or {}).items()}
    text = f"{req.get('title', '')}. {req.get('description', '')}"
    warnings = []

    started = clock()
    deadline = started + ROUTE_BUDGET
    hits = search_chunks(host, pinecone_key, req["namespace"], text, timeout=min(15, ROUTE_BUDGET))
    if not hits:
        raise RouteError(f"no matching code; has namespace {req['namespace']} been indexed?")
    searched = clock()

    ai = llm_triage(llm, req.get("title", ""), req.get("description", ""), req.get("severity", ""),
                    hits, bool(req.get("ask_questions")), deadline)
    ranked, note, questions, ai_used = [], "", [], False
    if ai and "error" in ai:
        warnings.append(ai["error"] + "; used the search order instead")
    elif ai:
        ranked, note, questions, ai_used = ai["ranked"], ai["note"], ai["questions"], True
    triaged = clock()
    blame_seconds = max(MIN_STEP_SECONDS, min(12, time_left(deadline) - 1))

    files = pick_files(hits, ranked)
    branch_missing = []

    def blame_one(item):
        index, hit = item
        start, end = (int(x) for x in hit["lines"].split("-"))
        try:
            try:
                return index, blame_file(owner, repo, token, hit["path"], start, end, ref or None,
                                         blame_seconds), None
            except BranchMissing as err:
                branch_missing.append(str(err))
                return index, blame_file(owner, repo, token, hit["path"], start, end, None,
                                         blame_seconds), None
        except (RouteError, requests.RequestException, ValueError) as err:
            return index, {}, str(err)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(blame_one, enumerate(files)))
    blamed = clock()
    if branch_missing:
        warnings.append(f"{branch_missing[0]}; used the default branch for blame")

    linked_emails = set(people.values())
    scores, debug_files, blamed_lines = {}, [], 0
    for index, owners, error in results:
        if error:
            warnings.append(error)
        authors = []
        for key, info in owners.items():
            cliq = match_cliq(info, people, linked_emails)
            weighted = info.get("weighted", info["lines"])
            authors.append({"key": key, "login": info["login"], "email": info["email"],
                            "lines": info["lines"], "weighted": round(weighted, 1), "cliq": cliq})
            blamed_lines += info["lines"]
            merge_key = cliq or key
            cand = scores.setdefault(merge_key, {"key": key, "name": info["name"], "login": info["login"],
                                                 "email": info["email"], "cliq": cliq, "score": 0,
                                                 "files": []})
            cand["score"] += weighted * FILE_WEIGHTS[index]
            if files[index]["path"] not in cand["files"]:
                cand["files"].append(files[index]["path"])
        debug_files.append({"path": files[index]["path"], "lines": files[index]["lines"],
                            "authors": authors, "error": error or ""})

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

    ranked_all = sorted(scores.values(), key=lambda c: -c["score"])
    for cand in ranked_all:
        cand["score"] = round(cand["score"], 1)
        if cand["score"] == int(cand["score"]):
            cand["score"] = int(cand["score"])
    where = code_confidence(ai)
    thin = rate_candidates(ranked_all, where["percent"], blamed_lines)
    candidates = ranked_all[:MAX_CANDIDATES]
    for cand in candidates:
        cand["codeowner"] = bool(cand.get("codeowner"))
    return {
        "files": [{"path": f["path"], "lines": f["lines"],
                   "url": f"https://github.com/{repo_full}/blob/{ref or 'HEAD'}/{f['path']}#L"
                          + f["lines"].replace("-", "-L")} for f in files],
        "candidates": candidates,
        "note": note,
        "questions": questions,
        "ai_used": ai_used,
        "ai_model": ai.get("model", "") if (ai and "error" not in ai) else "",
        "warnings": warnings,
        "code_confidence": where,
        "blamed_lines": blamed_lines,
        "thin_evidence": thin,
        "debug": {
            "ref": ref or "(default branch)",
            "token": "set" if token else "missing",
            "linked_people": sorted(people.keys()),
            "hits": [f"{h['path']}:{h['lines']}" for h in hits],
            "ranked": ranked,
            "codeowners": {"source": owners_source, "files": owners_by_file},
            "files": debug_files,
            "seconds": {"budget": ROUTE_BUDGET, "search": round(searched - started, 2), "ai": round(triaged - searched, 2),
                        "blame": round(blamed - triaged, 2), "total": round(blamed - started, 2)},
        },
    }
