// ---- router.py ----
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
// ---- app.py ----
"""
Conduit indexer service.

A small web service that does Conduit's heavy lifting:
    POST /index   index a GitHub repo into Pinecone (Add repo, Re-index)
    GET  /status  indexing progress for the /repo card
    POST /route   decide which code a bug is about and who should own it (/bug)
    POST /tickets/add, /tickets/remove, /tickets/similar, /tickets/clear
                  open tickets as searchable text, to catch duplicate reports
    POST /register
                  give a newly installed org its own secret
    POST /github/<key>
                  GitHub push webhook: re-index only the files a push changed
    POST /reminders/connect, /reminders/disconnect
                  an org hands over (or withdraws) an extension token so the
                  indexer can tick its copy of Conduit for reminders
Each org's secret carries a tenant id, and every namespace it touches is stored
under that tenant, so two orgs with a project called "payments" never mix.
It also "ticks" every org that turned on reminders, so each one's copy of
Conduit can send them.

Environment variables
    CONDUIT_SECRET      required. Any long random string. Signs every org's secret.
                        Used directly as a secret it acts as the original org,
                        whose data has no tenant prefix.
    CONDUIT_REGISTER_KEY optional. Turns on /register; the same value is built into
                        the extension, so only Conduit itself can sign orgs up.
    PINECONE_API_KEY    required. Your Pinecone API key.
    PINECONE_INDEX      optional. Index name, default "conduit".
    CLIQ_WEBHOOK_TOKEN  optional. A Cliq webhook token; when set, the service
                        messages the person through the bot when indexing ends.
    CLIQ_BOT_NAME       optional. Bot unique name, default "conduit".
    CLIQ_APPKEYS        App key of the Conduit extension (My Extensions -> Conduit ->
                        Connectors), used with CLIQ_WEBHOOK_TOKEN for the
                        "indexing finished" message.
    CLIQ_API_BASE       optional. Default https://cliq.zoho.com; India accounts
                        use https://cliq.zoho.in
    TICK_MINUTES        optional. How often to tick connected orgs while awake,
                        default 10. 0 turns it off (Catalyst Cron calls /tick anyway).
    LLM_API_KEY         optional. Turns on AI triage. A Groq key (gsk_...) by default.
    LLM_BASE_URL        optional. Any OpenAI-compatible endpoint; default Groq,
                        https://api.groq.com/openai/v1
    LLM_MODEL           optional. Default openai/gpt-oss-120b
    QUICKML_URL, QUICKML_ENDPOINT_KEY, QUICKML_ORG
                        optional. A Catalyst QuickML LLM endpoint; when set (with the
                        three ZOHO_* values) it is tried first and LLM_* is the fallback.
    QUICKML_ENVIRONMENT optional. Development (default) or Production
    QUICKML_MODEL_LABEL optional. Name shown in /repo, default GLM-4.7-Flash
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN
                        a Zoho Self Client with scope QuickML.deployment.READ
    ZOHO_ACCOUNTS_URL   optional. Default https://accounts.zoho.com
    CONDUIT_PUBLIC_URL  optional. This service's public https address, used for the
                        GitHub webhooks it creates; by default taken from the request.
    REINDEX_DELAY       optional. Seconds to wait after the last push before updating
                        the index, default 120, so a burst of pushes updates it once.
    ROUTE_BUDGET        optional. Seconds /route may take in total, default 20, so /bug
                        answers before Cliq gives up; the AI is skipped first.
    DUP_THRESHOLD       optional. How similar a new bug must be to an open ticket
                        to count as a duplicate, default 0.87

Runs on Zoho Catalyst AppSail (reads X_ZOHO_CATALYST_LISTEN_PORT) or any host
that sets PORT, such as Render.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request, send_from_directory

import index_repo
import quickml
import registry
import repos
import router
import tickets

app = Flask(__name__)

SECRET = os.environ.get("CONDUIT_SECRET", "")
REGISTER_KEY = os.environ.get("CONDUIT_REGISTER_KEY", "")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "conduit")
CLIQ_TOKEN = os.environ.get("CLIQ_WEBHOOK_TOKEN", "")
CLIQ_BOT = os.environ.get("CLIQ_BOT_NAME", "conduit")
CLIQ_BASE = os.environ.get("CLIQ_API_BASE", "https://cliq.zoho.com").rstrip("/")
CLIQ_APPKEYS = [k.strip() for k in os.environ.get("CLIQ_APPKEYS", "").split(",") if k.strip()]
TICK_MINUTES = float(os.environ.get("TICK_MINUTES", "10") or 0)
LLM = {
    "key": os.environ.get("LLM_API_KEY", ""),
    "base": os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    "model": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
    # Catalyst QuickML (Zoho-hosted model); tried first when configured
    "quickml": {
        "url": os.environ.get("QUICKML_URL", ""),
        "endpoint_key": os.environ.get("QUICKML_ENDPOINT_KEY", ""),
        "org": os.environ.get("QUICKML_ORG", ""),
        "environment": os.environ.get("QUICKML_ENVIRONMENT", "Development"),
        "label": os.environ.get("QUICKML_MODEL_LABEL", "GLM-4.7-Flash"),
        "client_id": os.environ.get("ZOHO_CLIENT_ID", ""),
        "client_secret": os.environ.get("ZOHO_CLIENT_SECRET", ""),
        "refresh_token": os.environ.get("ZOHO_REFRESH_TOKEN", ""),
        "accounts": os.environ.get("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.com"),
    },
}

PUBLIC_URL = os.environ.get("CONDUIT_PUBLIC_URL", "").rstrip("/")
REINDEX_DELAY = float(os.environ.get("REINDEX_DELAY", "120") or 120)
DUP_THRESHOLD = float(os.environ.get("DUP_THRESHOLD", tickets.DEFAULT_THRESHOLD) or tickets.DEFAULT_THRESHOLD)

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# a project may not end in --b: that name is its second index slot (see repos.py)
NAMESPACE_RE = re.compile(r"^(?!.*--b$)[a-z0-9_-]{1,64}$")
TOKEN_RE = re.compile(r"^cdt_(o[a-z0-9]{9})_([a-f0-9]{32})$")
ENDPOINT_RE = re.compile(r"^https://([a-z.]+)/api/v2/applications/\d+/incoming\?appkey=[A-Za-z0-9+/=_%-]+$")
# Cliq data centres; an org's token only works on its own one
CLIQ_HOSTS = ["cliq.zoho.com", "cliq.zoho.in", "cliq.zoho.eu", "cliq.zoho.com.au",
              "cliq.zoho.jp", "cliq.zohocloud.ca", "cliq.zoho.sa", "cliq.zoho.com.cn"]

jobs = {}             # namespace -> job dict
jobs_lock = threading.Lock()
pending = {}          # repo key -> files waiting for the push delay to pass
pending_lock = threading.Lock()
host_cache = {}       # index name -> host


def now():
    return datetime.now(timezone.utc)


def stamp(moment):
    return moment.strftime("%d %b %H:%M UTC")


def sign(tenant):
    return hmac.new(SECRET.encode(), tenant.encode(), hashlib.sha256).hexdigest()[:32]


def tenant_of(given):
    """The tenant a secret belongs to: "" for the original org, None if it is not valid."""
    if not SECRET or not given:
        return None
    if hmac.compare_digest(given, SECRET):
        return ""
    m = TOKEN_RE.match(given)
    if m and hmac.compare_digest(m.group(2), sign(m.group(1))):
        return m.group(1)
    return None


def caller():
    return tenant_of(request.headers.get("X-Conduit-Secret", ""))


@app.before_request
def guard_registry():
    """No project may be called like the registry namespace."""
    ns = request.args.get("namespace", "")
    if not ns and request.is_json:
        ns = str((request.get_json(silent=True) or {}).get("namespace", ""))
    if ns.startswith("__"):
        return jsonify({"error": "namespace must not start with __"}), 400


def scoped(tenant, namespace):
    """Where a project's data lives in Pinecone for this tenant."""
    return f"{tenant}-{namespace}" if tenant else namespace


def get_host():
    if INDEX_NAME not in host_cache:
        host_cache[INDEX_NAME] = index_repo.ensure_index(PINECONE_KEY, INDEX_NAME)
    return host_cache[INDEX_NAME]


def notify(email, text):
    """Message one person through the Cliq bot. Best effort only."""
    if not (CLIQ_TOKEN and email):
        return
    params = {"zapikey": CLIQ_TOKEN}
    if CLIQ_APPKEYS:
        params["appkey"] = CLIQ_APPKEYS[0]
    try:
        requests.post(f"{CLIQ_BASE}/api/v2/bots/{CLIQ_BOT}/message",
                      params=params, json={"text": text, "userids": email}, timeout=20)
    except requests.RequestException:
        pass


def summary(job):
    state = job["state"]
    if state == "running":
        if job["stage"] == "upload" and job["total"]:
            pct = int(100 * job["done"] / job["total"])
            return f"indexing, {pct}% ({job['done']}/{job['total']} chunks)"
        if job["stage"] == "upload":
            return f"indexing, {job['done']} chunks uploaded so far"
        return "indexing, downloading the repo"
    if state == "ready" and job.get("kind") == "update":
        return (f"updated {job['files']} changed files after a push, "
                f"finished {stamp(job['finished'])}")
    if state == "ready":
        return (f"indexed, {job['chunks']} chunks from {job['files']} files, "
                f"finished {stamp(job['finished'])}")
    return f"indexing failed: {job['error']}"


def public(job, namespace):
    return {
        "namespace": namespace,
        "state": job["state"],
        "summary": summary(job),
        "repo": job["repo"],
        "files": job.get("files"),
        "chunks": job.get("chunks"),
        "error": job.get("error"),
    }


def worker(key, namespace, repo, token, ref, email, rkey=None):
    """key is the project's data in Pinecone; namespace is the name the org knows.

    The new index is built in the project's other slot while /route keeps using the
    active one, then the project switches over and the old slot is cleared.
    """
    job = jobs[key]

    def progress(stage, done, total):
        job["stage"], job["done"], job["total"] = stage, done, total

    try:
        record = (repos.load(get_host(), PINECONE_KEY, SECRET, rkey) if rkey else None) or {}
        active = record.get("slot", "a")
        target = repos.other_slot(active)
        files, chunks, host = index_repo.run_index(
            repo, repos.physical(key, target), token, PINECONE_KEY, INDEX_NAME, ref, progress)
        host_cache[INDEX_NAME] = host
        if rkey:
            record = repos.load(host, PINECONE_KEY, SECRET, rkey) or record
            record.update(slot=target, indexed_at=int(time.time()))
            repos.save(host, PINECONE_KEY, SECRET, rkey, record)
            index_repo.clear_namespace(host, PINECONE_KEY, repos.physical(key, active))
        job.update(state="ready", files=files, chunks=chunks, finished=now())
        notify(email, f"✅ Indexing finished for *{namespace}* ({repo}): "
                      f"{chunks} chunks from {files} files. /bug can now search it.")
    except index_repo.IndexerError as err:
        job.update(state="failed", error=str(err), finished=now())
        notify(email, f"⚠️ Indexing failed for *{namespace}* ({repo}): {err}")
    except Exception as err:  # keep the service alive whatever happens
        job.update(state="failed", error=f"unexpected error: {err}", finished=now())
        notify(email, f"⚠️ Indexing failed for *{namespace}* ({repo}) with an unexpected error.")


def ai_label():
    names = []
    if quickml.configured(LLM["quickml"]):
        names.append(LLM["quickml"]["label"])
    if LLM["key"]:
        names.append(LLM["model"])
    return " → ".join(names) if names else "off"


@app.get("/")
def health():
    problems = []
    if not SECRET:
        problems.append("CONDUIT_SECRET is not set")
    if not PINECONE_KEY:
        problems.append("PINECONE_API_KEY is not set")
    return jsonify({
        "service": "conduit-indexer", "ok": not problems, "problems": problems,
        "ai": ai_label(),
        "registration": "open" if (SECRET and REGISTER_KEY) else "off",
        "reminders": f"every {TICK_MINUTES:g} min" if TICK_MINUTES > 0 else "on demand",
    })


@app.post("/index")
def start_index():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    repo = str(body.get("repo", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    token = str(body.get("token", "")).strip()
    ref = str(body.get("ref", "")).strip() or None
    email = str(body.get("notify", "")).strip()
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)

    rkey = repos.repo_key(tenant, namespace)

    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify(public(current, namespace)), 409
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": repo, "started": now()}
    try:
        auto = remember_repo(rkey, tenant, namespace, repo, token, ref)
    except (index_repo.IndexerError, requests.RequestException) as err:
        auto = {"state": "off", "reason": f"could not save the repo settings: {err.__class__.__name__}"}
        rkey = None
    threading.Thread(target=worker, args=(key, namespace, repo, token, ref, email, rkey),
                     daemon=True).start()
    out = public(jobs[key], namespace)
    out["auto_update"] = auto
    return jsonify(out), 202


def public_base():
    if PUBLIC_URL:
        return PUBLIC_URL
    proto = request.headers.get("X-Forwarded-Proto", "https")
    return f"{proto}://{request.host}".replace("http://", "https://")


def remember_repo(rkey, tenant, namespace, repo, token, ref):
    """Store the repo for automatic updates and make sure GitHub has a webhook for it.
    Returns {"state": "on"} or {"state": "off", "reason": ...} for the admin."""
    host = get_host()
    record = repos.load(host, PINECONE_KEY, SECRET, rkey) or {}
    if record.get("hook_id") and record.get("repo") != repo:
        repos.delete_hook(record.get("repo", ""), record.get("token", ""), record.get("hook_id"))
        record.pop("hook_id", None)
    record.update(tenant=tenant, namespace=namespace, repo=repo, ref=ref or "")
    if token:
        record["token"] = token
    auto = {"state": "on"}
    if not record.get("hook_id"):
        record["hook_secret"] = repos.new_hook_secret()
        hook_id, reason = repos.create_hook(repo, record.get("token", ""),
                                            f"{public_base()}/github/{rkey}", record["hook_secret"])
        if hook_id:
            record["hook_id"] = hook_id
        else:
            auto = {"state": "off", "reason": reason}
    repos.save(host, PINECONE_KEY, SECRET, rkey, record)
    return auto


@app.post("/github/<rkey>")
def github_push(rkey):
    """GitHub calls this on every push to a repo Conduit created a webhook for."""
    if not re.fullmatch(r"[a-f0-9]{24}", rkey) or not (PINECONE_KEY and SECRET):
        return jsonify({"error": "unknown hook"}), 404
    try:
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
    except (index_repo.IndexerError, requests.RequestException):
        return jsonify({"error": "could not read the repo settings"}), 502
    if not record:
        return jsonify({"error": "unknown hook"}), 404
    if not repos.signature_ok(record.get("hook_secret", ""), request.get_data(),
                              request.headers.get("X-Hub-Signature-256", "")):
        return jsonify({"error": "bad signature"}), 401
    event = request.headers.get("X-GitHub-Event", "")
    if event == "ping":
        return jsonify({"ok": True})
    if event != "push":
        return jsonify({"ignored": event}), 202
    change = repos.read_push(request.get_json(silent=True) or {}, record.get("ref", ""))
    if change is None:
        return jsonify({"ignored": "not the tracked branch"}), 202
    schedule_update(rkey, change)
    return jsonify({"queued": len(change["changed"]) + len(change["removed"]), "full": change["full"]}), 202


def schedule_update(rkey, change, delay=None):
    """Collect pushes and update the index once they stop for REINDEX_DELAY seconds."""
    with pending_lock:
        waiting = pending.setdefault(rkey, {"full": False, "changed": set(), "removed": set(), "timer": None})
        for path in change["changed"]:
            waiting["changed"].add(path)
            waiting["removed"].discard(path)
        for path in change["removed"]:
            waiting["removed"].add(path)
            waiting["changed"].discard(path)
        waiting["full"] = waiting["full"] or change["full"]
        if waiting["timer"]:
            waiting["timer"].cancel()
        timer = threading.Timer(REINDEX_DELAY if delay is None else delay, apply_update, args=(rkey,))
        timer.daemon = True
        waiting["timer"] = timer
        timer.start()


def apply_update(rkey):
    """Bring one project's index up to date with what was pushed."""
    with pending_lock:
        waiting = pending.pop(rkey, None)
    if not waiting:
        return
    try:
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
    except (index_repo.IndexerError, requests.RequestException):
        record = None
    if not record:
        return
    key = scoped(record["tenant"], record["namespace"])
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            # an index run is busy: try again a little later with the same changes
            change = {"full": waiting["full"], "changed": waiting["changed"], "removed": waiting["removed"]}
            threading.Thread(target=schedule_update, args=(rkey, change, 60), daemon=True).start()
            return
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": record["repo"], "started": now()}
    full = waiting["full"] or not record.get("indexed_at")
    if full:
        worker(key, record["namespace"], record["repo"], record.get("token", ""),
               record.get("ref") or None, "", rkey)
        return
    job = jobs[key]
    job["stage"] = "update"
    try:
        target = repos.physical(key, record.get("slot", "a"))
        updated, gone, chunks = index_repo.update_files(
            get_host(), PINECONE_KEY, target, record["repo"], record.get("token", ""),
            record.get("ref") or None, waiting["changed"], waiting["removed"])
        record.update(updated_at=int(time.time()), updated_files=updated + gone)
        repos.save(get_host(), PINECONE_KEY, SECRET, rkey, record)
        total = index_repo.namespace_count(get_host(), PINECONE_KEY, target)
        job.update(state="ready", kind="update", files=updated + gone, chunks=total or chunks,
                   finished=now())
    except (index_repo.IndexerError, requests.RequestException) as err:
        job.update(state="failed", error=f"update after a push failed: {err}", finished=now())
    except Exception as err:  # keep the service alive whatever happens
        job.update(state="failed", error=f"unexpected error: {err}", finished=now())


@app.post("/forget")
def forget_repo():
    """Delete a project's chunks from Pinecone when it is removed from Conduit."""
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    namespace = str(body.get("namespace", "")).strip()
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify({"error": "this project is being indexed right now"}), 409
        jobs.pop(key, None)
    rkey = repos.repo_key(tenant, namespace)
    with pending_lock:
        waiting = pending.pop(rkey, None)
        if waiting and waiting["timer"]:
            waiting["timer"].cancel()
    try:
        ok = index_repo.clear_namespace(get_host(), PINECONE_KEY, repos.physical(key, "a"))
        ok = index_repo.clear_namespace(get_host(), PINECONE_KEY, repos.physical(key, "b")) and ok
        tickets.forget(get_host(), PINECONE_KEY, key)
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
        if record:
            repos.delete_hook(record.get("repo", ""), record.get("token", ""), record.get("hook_id"))
            repos.remove(get_host(), PINECONE_KEY, rkey)
    except (index_repo.IndexerError, requests.RequestException) as err:
        return jsonify({"error": f"could not reach Pinecone: {err.__class__.__name__}"}), 502
    if not ok:
        return jsonify({"error": "Pinecone refused to delete the data"}), 502
    return jsonify({"namespace": namespace, "deleted": True})


@app.post("/route")
def route_bug():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    repo = str(body.get("repo", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    people = body.get("people") or {}
    if isinstance(people, str):
        try:
            people = json.loads(people)
        except ValueError:
            people = {}
    try:
        search_ns = repos.active_namespace(get_host(), PINECONE_KEY, SECRET, scoped(tenant, namespace),
                                           tenant, namespace)
    except (index_repo.IndexerError, requests.RequestException):
        search_ns = scoped(tenant, namespace)
    req = {
        "repo": repo, "namespace": search_ns, "token": str(body.get("token", "")).strip(),
        "title": str(body.get("title", ""))[:300], "description": str(body.get("description", ""))[:3000],
        "severity": str(body.get("severity", "")), "people": people,
        "ask_questions": str(body.get("ask_questions", "")).lower() in ("true", "1", "yes"),
        "ref": str(body.get("ref", "")).strip(),
    }
    try:
        result = router.route(req, get_host(), PINECONE_KEY, LLM)
    except (router.RouteError, index_repo.IndexerError) as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=str(err))
        return jsonify({"error": str(err)}), 422
    except requests.RequestException as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=f"network error: {err.__class__.__name__}")
        return jsonify({"error": f"network error: {err.__class__.__name__}"}), 502
    log_route(tenant, repo, namespace, req["ref"], people, result=result)
    return jsonify(result)


def log_route(tenant, repo, namespace, ref, people, result=None, error=""):
    """One line per /route call in the Catalyst logs: what was searched, blamed and matched."""
    line = {"event": "route", "tenant": tenant, "repo": repo, "namespace": namespace,
            "ref": ref or "(default branch)", "linked_people": len(people or {})}
    if error:
        line["error"] = error
    if result:
        line["debug"] = result.get("debug", {})
        line["candidates"] = [{"key": c["key"], "cliq": c["cliq"], "score": c["score"],
                               "share": c.get("share"), "confidence": c.get("confidence")}
                              for c in result.get("candidates", [])]
        line["code_confidence"] = result.get("code_confidence")
        line["warnings"] = result.get("warnings", [])
        line["ai"] = result.get("ai_model") or "search order"
    print("CONDUIT " + json.dumps(line, default=str), flush=True)


def ticket_request():
    """Common checks for the /tickets endpoints. Returns (body, error response)."""
    tenant = caller()
    if tenant is None:
        return None, (jsonify({"error": "wrong or missing secret"}), 401)
    if not PINECONE_KEY:
        return None, (jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500)
    body = request.get_json(silent=True) or {}
    namespace = str(body.get("namespace", "")).strip()
    if not NAMESPACE_RE.match(namespace):
        return None, (jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400)
    body["namespace"] = scoped(tenant, namespace)
    return body, None


@app.post("/tickets/add")
def tickets_add():
    body, err = ticket_request()
    if err:
        return err
    record_id = str(body.get("record_id", "")).strip()
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400
    try:
        tickets.add(get_host(), PINECONE_KEY, body["namespace"], record_id,
                    str(body.get("ticket_id", "")), str(body.get("title", ""))[:300],
                    str(body.get("text", ""))[:3000])
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not store the ticket: {e.__class__.__name__}"}), 502
    return jsonify({"stored": record_id})


@app.post("/tickets/remove")
def tickets_remove():
    body, err = ticket_request()
    if err:
        return err
    record_id = str(body.get("record_id", "")).strip()
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400
    try:
        ok = tickets.remove(get_host(), PINECONE_KEY, body["namespace"], record_id)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"removed": record_id, "ok": ok})


@app.post("/tickets/clear")
def tickets_clear():
    """Forget every stored ticket of one project (used by /sweep). The code index is kept."""
    body, err = ticket_request()
    if err:
        return err
    try:
        ok = tickets.forget(get_host(), PINECONE_KEY, body["namespace"])
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"cleared": bool(ok)})


@app.post("/tickets/similar")
def tickets_similar():
    body, err = ticket_request()
    if err:
        return err
    info = {}
    try:
        matches = tickets.similar(get_host(), PINECONE_KEY, body["namespace"],
                                  str(body.get("title", ""))[:300], str(body.get("text", ""))[:3000],
                                  DUP_THRESHOLD, body.get("exclude"), info)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not search tickets: {e.__class__.__name__}", "matches": []}), 502
    # "best" is the closest score seen, even below the threshold; handy for tuning DUP_THRESHOLD
    return jsonify({"matches": matches, "threshold": DUP_THRESHOLD, "best": info.get("best", 0)})


def ping_org(endpoint, token):
    """Tick one org's copy of Conduit through its extension incoming webhook."""
    try:
        r = requests.post(endpoint, params={"zapikey": token}, json={"action": "tick"}, timeout=30)
        return r.status_code
    except requests.RequestException:
        return "error"


def ping_orgs():
    """Tick every org that turned on reminders. Returns (orgs, delivered, error)."""
    if not (PINECONE_KEY and SECRET):
        return 0, 0, "the indexer has no PINECONE_API_KEY"
    try:
        orgs = registry.load(get_host(), PINECONE_KEY, SECRET)
    except (index_repo.IndexerError, requests.RequestException) as err:
        return 0, 0, f"could not read the org list: {err}"
    ok = 0
    for org in orgs:
        if ping_org(org["endpoint"], org["token"]) in (200, 204):
            ok += 1
    return len(orgs), ok, None


def ticker():
    """Ping the bots every TICK_MINUTES while the service is awake."""
    while True:
        time.sleep(TICK_MINUTES * 60)
        ping_orgs()


if SECRET and TICK_MINUTES > 0:
    threading.Thread(target=ticker, daemon=True).start()


@app.get("/flux.png")
def flux_icon():
    """Flux's avatar, public so Cliq can show it on messages Conduit posts in channels."""
    resp = send_from_directory(os.path.dirname(os.path.abspath(__file__)), "flux.png", mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.post("/register")
def register():
    """A new org installed Conduit: hand it a secret of its own."""
    given = request.headers.get("X-Conduit-Register", "")
    if not (SECRET and REGISTER_KEY) or not hmac.compare_digest(given, REGISTER_KEY):
        return jsonify({"error": "registration is not open"}), 401
    tenant = "o" + "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(9))
    return jsonify({"secret": f"cdt_{tenant}_{sign(tenant)}"})


@app.get("/privacy")
def privacy_page():
    """Public privacy policy, linked from the Zoho Marketplace listing."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal"), "privacy.html", mimetype="text/html")


@app.get("/terms")
def terms_page():
    """Public terms of service, linked from the Zoho Marketplace listing."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal"), "terms.html", mimetype="text/html")


@app.route("/tick", methods=["GET", "POST"])
def tick():
    """For an outside scheduler: keeps the service awake and sends reminders now."""
    given = request.args.get("secret", "") or request.headers.get("X-Conduit-Secret", "")
    if not SECRET or not hmac.compare_digest(given, SECRET):
        return jsonify({"error": "wrong or missing secret"}), 401
    orgs, delivered, error = ping_orgs()
    out = {"orgs": orgs, "delivered": delivered}
    if error:
        out["error"] = error
    return jsonify(out)


@app.post("/reminders/connect")
def reminders_connect():
    """Store an org's extension token, after checking it really reaches their Conduit."""
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    endpoint = str(body.get("endpoint", "")).strip()
    token = str(body.get("token", "")).strip()
    m = ENDPOINT_RE.match(endpoint)
    if not m or m.group(1) not in CLIQ_HOSTS:
        return jsonify({"error": "endpoint must be the extension's incoming webhook URL"}), 400
    given_host = m.group(1)
    if not token or len(token) > 500:
        return jsonify({"error": "token is missing"}), 400
    tried = {}
    # the org's own data centre first, then the others
    for host in [given_host] + [h for h in CLIQ_HOSTS if h != given_host]:
        candidate = endpoint.replace(given_host, host, 1)
        code = ping_org(candidate, token)
        tried[host] = code
        if code in (200, 204):
            try:
                registry.save(get_host(), PINECONE_KEY, SECRET, tenant, candidate, token)
            except (index_repo.IndexerError, requests.RequestException) as e:
                return jsonify({"error": f"could not save: {e.__class__.__name__}"}), 502
            return jsonify({"connected": True, "host": host})
    return jsonify({"error": "Cliq did not accept the token", "tried": tried}), 422


@app.post("/reminders/disconnect")
def reminders_disconnect():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    try:
        ok = registry.remove(get_host(), PINECONE_KEY, tenant)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"disconnected": ok})


@app.get("/status")
def status():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    namespace = request.args.get("namespace", "").strip()
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)
    record = None
    if PINECONE_KEY:
        try:
            record = repos.load(get_host(), PINECONE_KEY, SECRET, repos.repo_key(tenant, namespace))
        except (index_repo.IndexerError, requests.RequestException):
            record = None
    auto = {"state": "on" if (record and record.get("hook_id")) else "off",
            "updated_at": (record or {}).get("updated_at"), "updated_files": (record or {}).get("updated_files")}
    job = jobs.get(key)
    if job:
        out = public(job, namespace)
        out["auto_update"] = auto
        return jsonify(out)
    # The service may have restarted since the last run; ask Pinecone instead.
    count = None
    if PINECONE_KEY:
        try:
            count = index_repo.namespace_count(get_host(), PINECONE_KEY,
                                               repos.physical(key, (record or {}).get("slot", "a")))
        except index_repo.IndexerError:
            count = None
    if count:
        text = f"indexed, {count} chunks"
        state = "ready"
    elif count == 0:
        text, state = "not indexed yet", "missing"
    else:
        text, state = "status unknown", "unknown"
    return jsonify({"namespace": namespace, "state": state, "summary": text, "chunks": count,
                    "auto_update": auto})


if __name__ == "__main__":
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT") or os.environ.get("PORT") or 9000)
    app.run(host="0.0.0.0", port=port, threaded=True)
// ---- router.py ----
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
// ---- app.py ----
"""
Conduit indexer service.

A small web service that does Conduit's heavy lifting:
    POST /index   index a GitHub repo into Pinecone (Add repo, Re-index)
    GET  /status  indexing progress for the /repo card
    POST /route   decide which code a bug is about and who should own it (/bug)
    POST /tickets/add, /tickets/remove, /tickets/similar, /tickets/clear
                  open tickets as searchable text, to catch duplicate reports
    POST /register
                  give a newly installed org its own secret
    POST /github/<key>
                  GitHub push webhook: re-index only the files a push changed
    POST /reminders/connect, /reminders/disconnect
                  an org hands over (or withdraws) an extension token so the
                  indexer can tick its copy of Conduit for reminders
Each org's secret carries a tenant id, and every namespace it touches is stored
under that tenant, so two orgs with a project called "payments" never mix.
It also "ticks" every org that turned on reminders, so each one's copy of
Conduit can send them.

Environment variables
    CONDUIT_SECRET      required. Any long random string. Signs every org's secret.
                        Used directly as a secret it acts as the original org,
                        whose data has no tenant prefix.
    CONDUIT_REGISTER_KEY optional. Turns on /register; the same value is built into
                        the extension, so only Conduit itself can sign orgs up.
    PINECONE_API_KEY    required. Your Pinecone API key.
    PINECONE_INDEX      optional. Index name, default "conduit".
    CLIQ_WEBHOOK_TOKEN  optional. A Cliq webhook token; when set, the service
                        messages the person through the bot when indexing ends.
    CLIQ_BOT_NAME       optional. Bot unique name, default "conduit".
    CLIQ_APPKEYS        App key of the Conduit extension (My Extensions -> Conduit ->
                        Connectors), used with CLIQ_WEBHOOK_TOKEN for the
                        "indexing finished" message.
    CLIQ_API_BASE       optional. Default https://cliq.zoho.com; India accounts
                        use https://cliq.zoho.in
    TICK_MINUTES        optional. How often to tick connected orgs while awake,
                        default 10. 0 turns it off (Catalyst Cron calls /tick anyway).
    LLM_API_KEY         optional. Turns on AI triage. A Groq key (gsk_...) by default.
    LLM_BASE_URL        optional. Any OpenAI-compatible endpoint; default Groq,
                        https://api.groq.com/openai/v1
    LLM_MODEL           optional. Default openai/gpt-oss-120b
    QUICKML_URL, QUICKML_ENDPOINT_KEY, QUICKML_ORG
                        optional. A Catalyst QuickML LLM endpoint; when set (with the
                        three ZOHO_* values) it is tried first and LLM_* is the fallback.
    QUICKML_ENVIRONMENT optional. Development (default) or Production
    QUICKML_MODEL_LABEL optional. Name shown in /repo, default GLM-4.7-Flash
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN
                        a Zoho Self Client with scope QuickML.deployment.READ
    ZOHO_ACCOUNTS_URL   optional. Default https://accounts.zoho.com
    CONDUIT_PUBLIC_URL  optional. This service's public https address, used for the
                        GitHub webhooks it creates; by default taken from the request.
    REINDEX_DELAY       optional. Seconds to wait after the last push before updating
                        the index, default 120, so a burst of pushes updates it once.
    ROUTE_BUDGET        optional. Seconds /route may take in total, default 20, so /bug
                        answers before Cliq gives up; the AI is skipped first.
    DUP_THRESHOLD       optional. How similar a new bug must be to an open ticket
                        to count as a duplicate, default 0.87

Runs on Zoho Catalyst AppSail (reads X_ZOHO_CATALYST_LISTEN_PORT) or any host
that sets PORT, such as Render.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request, send_from_directory

import index_repo
import quickml
import registry
import repos
import router
import tickets

app = Flask(__name__)

SECRET = os.environ.get("CONDUIT_SECRET", "")
REGISTER_KEY = os.environ.get("CONDUIT_REGISTER_KEY", "")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "conduit")
CLIQ_TOKEN = os.environ.get("CLIQ_WEBHOOK_TOKEN", "")
CLIQ_BOT = os.environ.get("CLIQ_BOT_NAME", "conduit")
CLIQ_BASE = os.environ.get("CLIQ_API_BASE", "https://cliq.zoho.com").rstrip("/")
CLIQ_APPKEYS = [k.strip() for k in os.environ.get("CLIQ_APPKEYS", "").split(",") if k.strip()]
TICK_MINUTES = float(os.environ.get("TICK_MINUTES", "10") or 0)
LLM = {
    "key": os.environ.get("LLM_API_KEY", ""),
    "base": os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    "model": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
    # Catalyst QuickML (Zoho-hosted model); tried first when configured
    "quickml": {
        "url": os.environ.get("QUICKML_URL", ""),
        "endpoint_key": os.environ.get("QUICKML_ENDPOINT_KEY", ""),
        "org": os.environ.get("QUICKML_ORG", ""),
        "environment": os.environ.get("QUICKML_ENVIRONMENT", "Development"),
        "label": os.environ.get("QUICKML_MODEL_LABEL", "GLM-4.7-Flash"),
        "client_id": os.environ.get("ZOHO_CLIENT_ID", ""),
        "client_secret": os.environ.get("ZOHO_CLIENT_SECRET", ""),
        "refresh_token": os.environ.get("ZOHO_REFRESH_TOKEN", ""),
        "accounts": os.environ.get("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.com"),
    },
}

PUBLIC_URL = os.environ.get("CONDUIT_PUBLIC_URL", "").rstrip("/")
REINDEX_DELAY = float(os.environ.get("REINDEX_DELAY", "120") or 120)
DUP_THRESHOLD = float(os.environ.get("DUP_THRESHOLD", tickets.DEFAULT_THRESHOLD) or tickets.DEFAULT_THRESHOLD)

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# a project may not end in --b: that name is its second index slot (see repos.py)
NAMESPACE_RE = re.compile(r"^(?!.*--b$)[a-z0-9_-]{1,64}$")
TOKEN_RE = re.compile(r"^cdt_(o[a-z0-9]{9})_([a-f0-9]{32})$")
ENDPOINT_RE = re.compile(r"^https://([a-z.]+)/api/v2/applications/\d+/incoming\?appkey=[A-Za-z0-9+/=_%-]+$")
# Cliq data centres; an org's token only works on its own one
CLIQ_HOSTS = ["cliq.zoho.com", "cliq.zoho.in", "cliq.zoho.eu", "cliq.zoho.com.au",
              "cliq.zoho.jp", "cliq.zohocloud.ca", "cliq.zoho.sa", "cliq.zoho.com.cn"]

jobs = {}             # namespace -> job dict
jobs_lock = threading.Lock()
pending = {}          # repo key -> files waiting for the push delay to pass
pending_lock = threading.Lock()
host_cache = {}       # index name -> host


def now():
    return datetime.now(timezone.utc)


def stamp(moment):
    return moment.strftime("%d %b %H:%M UTC")


def sign(tenant):
    return hmac.new(SECRET.encode(), tenant.encode(), hashlib.sha256).hexdigest()[:32]


def tenant_of(given):
    """The tenant a secret belongs to: "" for the original org, None if it is not valid."""
    if not SECRET or not given:
        return None
    if hmac.compare_digest(given, SECRET):
        return ""
    m = TOKEN_RE.match(given)
    if m and hmac.compare_digest(m.group(2), sign(m.group(1))):
        return m.group(1)
    return None


def caller():
    return tenant_of(request.headers.get("X-Conduit-Secret", ""))


@app.before_request
def guard_registry():
    """No project may be called like the registry namespace."""
    ns = request.args.get("namespace", "")
    if not ns and request.is_json:
        ns = str((request.get_json(silent=True) or {}).get("namespace", ""))
    if ns.startswith("__"):
        return jsonify({"error": "namespace must not start with __"}), 400


def scoped(tenant, namespace):
    """Where a project's data lives in Pinecone for this tenant."""
    return f"{tenant}-{namespace}" if tenant else namespace


def get_host():
    if INDEX_NAME not in host_cache:
        host_cache[INDEX_NAME] = index_repo.ensure_index(PINECONE_KEY, INDEX_NAME)
    return host_cache[INDEX_NAME]


def notify(email, text):
    """Message one person through the Cliq bot. Best effort only."""
    if not (CLIQ_TOKEN and email):
        return
    params = {"zapikey": CLIQ_TOKEN}
    if CLIQ_APPKEYS:
        params["appkey"] = CLIQ_APPKEYS[0]
    try:
        requests.post(f"{CLIQ_BASE}/api/v2/bots/{CLIQ_BOT}/message",
                      params=params, json={"text": text, "userids": email}, timeout=20)
    except requests.RequestException:
        pass


def summary(job):
    state = job["state"]
    if state == "running":
        if job["stage"] == "upload" and job["total"]:
            pct = int(100 * job["done"] / job["total"])
            return f"indexing, {pct}% ({job['done']}/{job['total']} chunks)"
        if job["stage"] == "upload":
            return f"indexing, {job['done']} chunks uploaded so far"
        return "indexing, downloading the repo"
    if state == "ready" and job.get("kind") == "update":
        return (f"updated {job['files']} changed files after a push, "
                f"finished {stamp(job['finished'])}")
    if state == "ready":
        return (f"indexed, {job['chunks']} chunks from {job['files']} files, "
                f"finished {stamp(job['finished'])}")
    return f"indexing failed: {job['error']}"


def public(job, namespace):
    return {
        "namespace": namespace,
        "state": job["state"],
        "summary": summary(job),
        "repo": job["repo"],
        "files": job.get("files"),
        "chunks": job.get("chunks"),
        "error": job.get("error"),
    }


def worker(key, namespace, repo, token, ref, email, rkey=None):
    """key is the project's data in Pinecone; namespace is the name the org knows.

    The new index is built in the project's other slot while /route keeps using the
    active one, then the project switches over and the old slot is cleared.
    """
    job = jobs[key]

    def progress(stage, done, total):
        job["stage"], job["done"], job["total"] = stage, done, total

    try:
        record = (repos.load(get_host(), PINECONE_KEY, SECRET, rkey) if rkey else None) or {}
        active = record.get("slot", "a")
        target = repos.other_slot(active)
        files, chunks, host = index_repo.run_index(
            repo, repos.physical(key, target), token, PINECONE_KEY, INDEX_NAME, ref, progress)
        host_cache[INDEX_NAME] = host
        if rkey:
            record = repos.load(host, PINECONE_KEY, SECRET, rkey) or record
            record.update(slot=target, indexed_at=int(time.time()))
            repos.save(host, PINECONE_KEY, SECRET, rkey, record)
            index_repo.clear_namespace(host, PINECONE_KEY, repos.physical(key, active))
        job.update(state="ready", files=files, chunks=chunks, finished=now())
        notify(email, f"✅ Indexing finished for *{namespace}* ({repo}): "
                      f"{chunks} chunks from {files} files. /bug can now search it.")
    except index_repo.IndexerError as err:
        job.update(state="failed", error=str(err), finished=now())
        notify(email, f"⚠️ Indexing failed for *{namespace}* ({repo}): {err}")
    except Exception as err:  # keep the service alive whatever happens
        job.update(state="failed", error=f"unexpected error: {err}", finished=now())
        notify(email, f"⚠️ Indexing failed for *{namespace}* ({repo}) with an unexpected error.")


def ai_label():
    names = []
    if quickml.configured(LLM["quickml"]):
        names.append(LLM["quickml"]["label"])
    if LLM["key"]:
        names.append(LLM["model"])
    return " → ".join(names) if names else "off"


@app.get("/")
def health():
    problems = []
    if not SECRET:
        problems.append("CONDUIT_SECRET is not set")
    if not PINECONE_KEY:
        problems.append("PINECONE_API_KEY is not set")
    return jsonify({
        "service": "conduit-indexer", "ok": not problems, "problems": problems,
        "ai": ai_label(),
        "registration": "open" if (SECRET and REGISTER_KEY) else "off",
        "reminders": f"every {TICK_MINUTES:g} min" if TICK_MINUTES > 0 else "on demand",
    })


@app.post("/index")
def start_index():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    repo = str(body.get("repo", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    token = str(body.get("token", "")).strip()
    ref = str(body.get("ref", "")).strip() or None
    email = str(body.get("notify", "")).strip()
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)

    rkey = repos.repo_key(tenant, namespace)

    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify(public(current, namespace)), 409
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": repo, "started": now()}
    try:
        auto = remember_repo(rkey, tenant, namespace, repo, token, ref)
    except (index_repo.IndexerError, requests.RequestException) as err:
        auto = {"state": "off", "reason": f"could not save the repo settings: {err.__class__.__name__}"}
        rkey = None
    threading.Thread(target=worker, args=(key, namespace, repo, token, ref, email, rkey),
                     daemon=True).start()
    out = public(jobs[key], namespace)
    out["auto_update"] = auto
    return jsonify(out), 202


def public_base():
    if PUBLIC_URL:
        return PUBLIC_URL
    proto = request.headers.get("X-Forwarded-Proto", "https")
    return f"{proto}://{request.host}".replace("http://", "https://")


def remember_repo(rkey, tenant, namespace, repo, token, ref):
    """Store the repo for automatic updates and make sure GitHub has a webhook for it.
    Returns {"state": "on"} or {"state": "off", "reason": ...} for the admin."""
    host = get_host()
    record = repos.load(host, PINECONE_KEY, SECRET, rkey) or {}
    if record.get("hook_id") and record.get("repo") != repo:
        repos.delete_hook(record.get("repo", ""), record.get("token", ""), record.get("hook_id"))
        record.pop("hook_id", None)
    record.update(tenant=tenant, namespace=namespace, repo=repo, ref=ref or "")
    if token:
        record["token"] = token
    auto = {"state": "on"}
    if not record.get("hook_id"):
        record["hook_secret"] = repos.new_hook_secret()
        hook_id, reason = repos.create_hook(repo, record.get("token", ""),
                                            f"{public_base()}/github/{rkey}", record["hook_secret"])
        if hook_id:
            record["hook_id"] = hook_id
        else:
            auto = {"state": "off", "reason": reason}
    repos.save(host, PINECONE_KEY, SECRET, rkey, record)
    return auto


@app.post("/github/<rkey>")
def github_push(rkey):
    """GitHub calls this on every push to a repo Conduit created a webhook for."""
    if not re.fullmatch(r"[a-f0-9]{24}", rkey) or not (PINECONE_KEY and SECRET):
        return jsonify({"error": "unknown hook"}), 404
    try:
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
    except (index_repo.IndexerError, requests.RequestException):
        return jsonify({"error": "could not read the repo settings"}), 502
    if not record:
        return jsonify({"error": "unknown hook"}), 404
    if not repos.signature_ok(record.get("hook_secret", ""), request.get_data(),
                              request.headers.get("X-Hub-Signature-256", "")):
        return jsonify({"error": "bad signature"}), 401
    event = request.headers.get("X-GitHub-Event", "")
    if event == "ping":
        return jsonify({"ok": True})
    if event != "push":
        return jsonify({"ignored": event}), 202
    change = repos.read_push(request.get_json(silent=True) or {}, record.get("ref", ""))
    if change is None:
        return jsonify({"ignored": "not the tracked branch"}), 202
    schedule_update(rkey, change)
    return jsonify({"queued": len(change["changed"]) + len(change["removed"]), "full": change["full"]}), 202


def schedule_update(rkey, change, delay=None):
    """Collect pushes and update the index once they stop for REINDEX_DELAY seconds."""
    with pending_lock:
        waiting = pending.setdefault(rkey, {"full": False, "changed": set(), "removed": set(), "timer": None})
        for path in change["changed"]:
            waiting["changed"].add(path)
            waiting["removed"].discard(path)
        for path in change["removed"]:
            waiting["removed"].add(path)
            waiting["changed"].discard(path)
        waiting["full"] = waiting["full"] or change["full"]
        if waiting["timer"]:
            waiting["timer"].cancel()
        timer = threading.Timer(REINDEX_DELAY if delay is None else delay, apply_update, args=(rkey,))
        timer.daemon = True
        waiting["timer"] = timer
        timer.start()


def apply_update(rkey):
    """Bring one project's index up to date with what was pushed."""
    with pending_lock:
        waiting = pending.pop(rkey, None)
    if not waiting:
        return
    try:
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
    except (index_repo.IndexerError, requests.RequestException):
        record = None
    if not record:
        return
    key = scoped(record["tenant"], record["namespace"])
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            # an index run is busy: try again a little later with the same changes
            change = {"full": waiting["full"], "changed": waiting["changed"], "removed": waiting["removed"]}
            threading.Thread(target=schedule_update, args=(rkey, change, 60), daemon=True).start()
            return
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": record["repo"], "started": now()}
    full = waiting["full"] or not record.get("indexed_at")
    if full:
        worker(key, record["namespace"], record["repo"], record.get("token", ""),
               record.get("ref") or None, "", rkey)
        return
    job = jobs[key]
    job["stage"] = "update"
    try:
        target = repos.physical(key, record.get("slot", "a"))
        updated, gone, chunks = index_repo.update_files(
            get_host(), PINECONE_KEY, target, record["repo"], record.get("token", ""),
            record.get("ref") or None, waiting["changed"], waiting["removed"])
        record.update(updated_at=int(time.time()), updated_files=updated + gone)
        repos.save(get_host(), PINECONE_KEY, SECRET, rkey, record)
        total = index_repo.namespace_count(get_host(), PINECONE_KEY, target)
        job.update(state="ready", kind="update", files=updated + gone, chunks=total or chunks,
                   finished=now())
    except (index_repo.IndexerError, requests.RequestException) as err:
        job.update(state="failed", error=f"update after a push failed: {err}", finished=now())
    except Exception as err:  # keep the service alive whatever happens
        job.update(state="failed", error=f"unexpected error: {err}", finished=now())


@app.post("/forget")
def forget_repo():
    """Delete a project's chunks from Pinecone when it is removed from Conduit."""
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    namespace = str(body.get("namespace", "")).strip()
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify({"error": "this project is being indexed right now"}), 409
        jobs.pop(key, None)
    rkey = repos.repo_key(tenant, namespace)
    with pending_lock:
        waiting = pending.pop(rkey, None)
        if waiting and waiting["timer"]:
            waiting["timer"].cancel()
    try:
        ok = index_repo.clear_namespace(get_host(), PINECONE_KEY, repos.physical(key, "a"))
        ok = index_repo.clear_namespace(get_host(), PINECONE_KEY, repos.physical(key, "b")) and ok
        tickets.forget(get_host(), PINECONE_KEY, key)
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
        if record:
            repos.delete_hook(record.get("repo", ""), record.get("token", ""), record.get("hook_id"))
            repos.remove(get_host(), PINECONE_KEY, rkey)
    except (index_repo.IndexerError, requests.RequestException) as err:
        return jsonify({"error": f"could not reach Pinecone: {err.__class__.__name__}"}), 502
    if not ok:
        return jsonify({"error": "Pinecone refused to delete the data"}), 502
    return jsonify({"namespace": namespace, "deleted": True})


@app.post("/route")
def route_bug():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    repo = str(body.get("repo", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    people = body.get("people") or {}
    if isinstance(people, str):
        try:
            people = json.loads(people)
        except ValueError:
            people = {}
    try:
        search_ns = repos.active_namespace(get_host(), PINECONE_KEY, SECRET, scoped(tenant, namespace),
                                           tenant, namespace)
    except (index_repo.IndexerError, requests.RequestException):
        search_ns = scoped(tenant, namespace)
    req = {
        "repo": repo, "namespace": search_ns, "token": str(body.get("token", "")).strip(),
        "title": str(body.get("title", ""))[:300], "description": str(body.get("description", ""))[:3000],
        "severity": str(body.get("severity", "")), "people": people,
        "ask_questions": str(body.get("ask_questions", "")).lower() in ("true", "1", "yes"),
        "ref": str(body.get("ref", "")).strip(),
    }
    try:
        result = router.route(req, get_host(), PINECONE_KEY, LLM)
    except (router.RouteError, index_repo.IndexerError) as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=str(err))
        return jsonify({"error": str(err)}), 422
    except requests.RequestException as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=f"network error: {err.__class__.__name__}")
        return jsonify({"error": f"network error: {err.__class__.__name__}"}), 502
    log_route(tenant, repo, namespace, req["ref"], people, result=result)
    return jsonify(result)


def log_route(tenant, repo, namespace, ref, people, result=None, error=""):
    """One line per /route call in the Catalyst logs: what was searched, blamed and matched."""
    line = {"event": "route", "tenant": tenant, "repo": repo, "namespace": namespace,
            "ref": ref or "(default branch)", "linked_people": len(people or {})}
    if error:
        line["error"] = error
    if result:
        line["debug"] = result.get("debug", {})
        line["candidates"] = [{"key": c["key"], "cliq": c["cliq"], "score": c["score"],
                               "share": c.get("share"), "confidence": c.get("confidence")}
                              for c in result.get("candidates", [])]
        line["code_confidence"] = result.get("code_confidence")
        line["warnings"] = result.get("warnings", [])
        line["ai"] = result.get("ai_model") or "search order"
    print("CONDUIT " + json.dumps(line, default=str), flush=True)


def ticket_request():
    """Common checks for the /tickets endpoints. Returns (body, error response)."""
    tenant = caller()
    if tenant is None:
        return None, (jsonify({"error": "wrong or missing secret"}), 401)
    if not PINECONE_KEY:
        return None, (jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500)
    body = request.get_json(silent=True) or {}
    namespace = str(body.get("namespace", "")).strip()
    if not NAMESPACE_RE.match(namespace):
        return None, (jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400)
    body["namespace"] = scoped(tenant, namespace)
    return body, None


@app.post("/tickets/add")
def tickets_add():
    body, err = ticket_request()
    if err:
        return err
    record_id = str(body.get("record_id", "")).strip()
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400
    try:
        tickets.add(get_host(), PINECONE_KEY, body["namespace"], record_id,
                    str(body.get("ticket_id", "")), str(body.get("title", ""))[:300],
                    str(body.get("text", ""))[:3000])
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not store the ticket: {e.__class__.__name__}"}), 502
    return jsonify({"stored": record_id})


@app.post("/tickets/remove")
def tickets_remove():
    body, err = ticket_request()
    if err:
        return err
    record_id = str(body.get("record_id", "")).strip()
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400
    try:
        ok = tickets.remove(get_host(), PINECONE_KEY, body["namespace"], record_id)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"removed": record_id, "ok": ok})


@app.post("/tickets/clear")
def tickets_clear():
    """Forget every stored ticket of one project (used by /sweep). The code index is kept."""
    body, err = ticket_request()
    if err:
        return err
    try:
        ok = tickets.forget(get_host(), PINECONE_KEY, body["namespace"])
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"cleared": bool(ok)})


@app.post("/tickets/similar")
def tickets_similar():
    body, err = ticket_request()
    if err:
        return err
    info = {}
    try:
        matches = tickets.similar(get_host(), PINECONE_KEY, body["namespace"],
                                  str(body.get("title", ""))[:300], str(body.get("text", ""))[:3000],
                                  DUP_THRESHOLD, body.get("exclude"), info)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not search tickets: {e.__class__.__name__}", "matches": []}), 502
    # "best" is the closest score seen, even below the threshold; handy for tuning DUP_THRESHOLD
    return jsonify({"matches": matches, "threshold": DUP_THRESHOLD, "best": info.get("best", 0)})


def ping_org(endpoint, token):
    """Tick one org's copy of Conduit through its extension incoming webhook."""
    try:
        r = requests.post(endpoint, params={"zapikey": token}, json={"action": "tick"}, timeout=30)
        return r.status_code
    except requests.RequestException:
        return "error"


def ping_orgs():
    """Tick every org that turned on reminders. Returns (orgs, delivered, error)."""
    if not (PINECONE_KEY and SECRET):
        return 0, 0, "the indexer has no PINECONE_API_KEY"
    try:
        orgs = registry.load(get_host(), PINECONE_KEY, SECRET)
    except (index_repo.IndexerError, requests.RequestException) as err:
        return 0, 0, f"could not read the org list: {err}"
    ok = 0
    for org in orgs:
        if ping_org(org["endpoint"], org["token"]) in (200, 204):
            ok += 1
    return len(orgs), ok, None


def ticker():
    """Ping the bots every TICK_MINUTES while the service is awake."""
    while True:
        time.sleep(TICK_MINUTES * 60)
        ping_orgs()


if SECRET and TICK_MINUTES > 0:
    threading.Thread(target=ticker, daemon=True).start()


@app.get("/flux.png")
def flux_icon():
    """Flux's avatar, public so Cliq can show it on messages Conduit posts in channels."""
    resp = send_from_directory(os.path.dirname(os.path.abspath(__file__)), "flux.png", mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.post("/register")
def register():
    """A new org installed Conduit: hand it a secret of its own."""
    given = request.headers.get("X-Conduit-Register", "")
    if not (SECRET and REGISTER_KEY) or not hmac.compare_digest(given, REGISTER_KEY):
        return jsonify({"error": "registration is not open"}), 401
    tenant = "o" + "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(9))
    return jsonify({"secret": f"cdt_{tenant}_{sign(tenant)}"})


@app.get("/privacy")
def privacy_page():
    """Public privacy policy, linked from the Zoho Marketplace listing."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal"), "privacy.html", mimetype="text/html")


@app.get("/terms")
def terms_page():
    """Public terms of service, linked from the Zoho Marketplace listing."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal"), "terms.html", mimetype="text/html")


@app.route("/tick", methods=["GET", "POST"])
def tick():
    """For an outside scheduler: keeps the service awake and sends reminders now."""
    given = request.args.get("secret", "") or request.headers.get("X-Conduit-Secret", "")
    if not SECRET or not hmac.compare_digest(given, SECRET):
        return jsonify({"error": "wrong or missing secret"}), 401
    orgs, delivered, error = ping_orgs()
    out = {"orgs": orgs, "delivered": delivered}
    if error:
        out["error"] = error
    return jsonify(out)


@app.post("/reminders/connect")
def reminders_connect():
    """Store an org's extension token, after checking it really reaches their Conduit."""
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    endpoint = str(body.get("endpoint", "")).strip()
    token = str(body.get("token", "")).strip()
    m = ENDPOINT_RE.match(endpoint)
    if not m or m.group(1) not in CLIQ_HOSTS:
        return jsonify({"error": "endpoint must be the extension's incoming webhook URL"}), 400
    given_host = m.group(1)
    if not token or len(token) > 500:
        return jsonify({"error": "token is missing"}), 400
    tried = {}
    # the org's own data centre first, then the others
    for host in [given_host] + [h for h in CLIQ_HOSTS if h != given_host]:
        candidate = endpoint.replace(given_host, host, 1)
        code = ping_org(candidate, token)
        tried[host] = code
        if code in (200, 204):
            try:
                registry.save(get_host(), PINECONE_KEY, SECRET, tenant, candidate, token)
            except (index_repo.IndexerError, requests.RequestException) as e:
                return jsonify({"error": f"could not save: {e.__class__.__name__}"}), 502
            return jsonify({"connected": True, "host": host})
    return jsonify({"error": "Cliq did not accept the token", "tried": tried}), 422


@app.post("/reminders/disconnect")
def reminders_disconnect():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    try:
        ok = registry.remove(get_host(), PINECONE_KEY, tenant)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"disconnected": ok})


@app.get("/status")
def status():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    namespace = request.args.get("namespace", "").strip()
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)
    record = None
    if PINECONE_KEY:
        try:
            record = repos.load(get_host(), PINECONE_KEY, SECRET, repos.repo_key(tenant, namespace))
        except (index_repo.IndexerError, requests.RequestException):
            record = None
    auto = {"state": "on" if (record and record.get("hook_id")) else "off",
            "updated_at": (record or {}).get("updated_at"), "updated_files": (record or {}).get("updated_files")}
    job = jobs.get(key)
    if job:
        out = public(job, namespace)
        out["auto_update"] = auto
        return jsonify(out)
    # The service may have restarted since the last run; ask Pinecone instead.
    count = None
    if PINECONE_KEY:
        try:
            count = index_repo.namespace_count(get_host(), PINECONE_KEY,
                                               repos.physical(key, (record or {}).get("slot", "a")))
        except index_repo.IndexerError:
            count = None
    if count:
        text = f"indexed, {count} chunks"
        state = "ready"
    elif count == 0:
        text, state = "not indexed yet", "missing"
    else:
        text, state = "status unknown", "unknown"
    return jsonify({"namespace": namespace, "state": state, "summary": text, "chunks": count,
                    "auto_update": auto})


if __name__ == "__main__":
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT") or os.environ.get("PORT") or 9000)
    app.run(host="0.0.0.0", port=port, threaded=True)
// ---- router.py ----
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
// ---- app.py ----
"""
Conduit indexer service.

A small web service that does Conduit's heavy lifting:
    POST /index   index a GitHub repo into Pinecone (Add repo, Re-index)
    GET  /status  indexing progress for the /repo card
    POST /route   decide which code a bug is about and who should own it (/bug)
    POST /tickets/add, /tickets/remove, /tickets/similar, /tickets/clear
                  open tickets as searchable text, to catch duplicate reports
    POST /register
                  give a newly installed org its own secret
    POST /github/<key>
                  GitHub push webhook: re-index only the files a push changed
    POST /reminders/connect, /reminders/disconnect
                  an org hands over (or withdraws) an extension token so the
                  indexer can tick its copy of Conduit for reminders
Each org's secret carries a tenant id, and every namespace it touches is stored
under that tenant, so two orgs with a project called "payments" never mix.
It also "ticks" every org that turned on reminders, so each one's copy of
Conduit can send them.

Environment variables
    CONDUIT_SECRET      required. Any long random string. Signs every org's secret.
                        Used directly as a secret it acts as the original org,
                        whose data has no tenant prefix.
    CONDUIT_REGISTER_KEY optional. Turns on /register; the same value is built into
                        the extension, so only Conduit itself can sign orgs up.
    PINECONE_API_KEY    required. Your Pinecone API key.
    PINECONE_INDEX      optional. Index name, default "conduit".
    CLIQ_WEBHOOK_TOKEN  optional. A Cliq webhook token; when set, the service
                        messages the person through the bot when indexing ends.
    CLIQ_BOT_NAME       optional. Bot unique name, default "conduit".
    CLIQ_APPKEYS        App key of the Conduit extension (My Extensions -> Conduit ->
                        Connectors), used with CLIQ_WEBHOOK_TOKEN for the
                        "indexing finished" message.
    CLIQ_API_BASE       optional. Default https://cliq.zoho.com; India accounts
                        use https://cliq.zoho.in
    TICK_MINUTES        optional. How often to tick connected orgs while awake,
                        default 10. 0 turns it off (Catalyst Cron calls /tick anyway).
    LLM_API_KEY         optional. Turns on AI triage. A Groq key (gsk_...) by default.
    LLM_BASE_URL        optional. Any OpenAI-compatible endpoint; default Groq,
                        https://api.groq.com/openai/v1
    LLM_MODEL           optional. Default openai/gpt-oss-120b
    QUICKML_URL, QUICKML_ENDPOINT_KEY, QUICKML_ORG
                        optional. A Catalyst QuickML LLM endpoint; when set (with the
                        three ZOHO_* values) it is tried first and LLM_* is the fallback.
    QUICKML_ENVIRONMENT optional. Development (default) or Production
    QUICKML_MODEL_LABEL optional. Name shown in /repo, default GLM-4.7-Flash
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN
                        a Zoho Self Client with scope QuickML.deployment.READ
    ZOHO_ACCOUNTS_URL   optional. Default https://accounts.zoho.com
    CONDUIT_PUBLIC_URL  optional. This service's public https address, used for the
                        GitHub webhooks it creates; by default taken from the request.
    REINDEX_DELAY       optional. Seconds to wait after the last push before updating
                        the index, default 120, so a burst of pushes updates it once.
    ROUTE_BUDGET        optional. Seconds /route may take in total, default 20, so /bug
                        answers before Cliq gives up; the AI is skipped first.
    DUP_THRESHOLD       optional. How similar a new bug must be to an open ticket
                        to count as a duplicate, default 0.87

Runs on Zoho Catalyst AppSail (reads X_ZOHO_CATALYST_LISTEN_PORT) or any host
that sets PORT, such as Render.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request, send_from_directory

import index_repo
import quickml
import registry
import repos
import router
import tickets

app = Flask(__name__)

SECRET = os.environ.get("CONDUIT_SECRET", "")
REGISTER_KEY = os.environ.get("CONDUIT_REGISTER_KEY", "")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "conduit")
CLIQ_TOKEN = os.environ.get("CLIQ_WEBHOOK_TOKEN", "")
CLIQ_BOT = os.environ.get("CLIQ_BOT_NAME", "conduit")
CLIQ_BASE = os.environ.get("CLIQ_API_BASE", "https://cliq.zoho.com").rstrip("/")
CLIQ_APPKEYS = [k.strip() for k in os.environ.get("CLIQ_APPKEYS", "").split(",") if k.strip()]
TICK_MINUTES = float(os.environ.get("TICK_MINUTES", "10") or 0)
LLM = {
    "key": os.environ.get("LLM_API_KEY", ""),
    "base": os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    "model": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
    # Catalyst QuickML (Zoho-hosted model); tried first when configured
    "quickml": {
        "url": os.environ.get("QUICKML_URL", ""),
        "endpoint_key": os.environ.get("QUICKML_ENDPOINT_KEY", ""),
        "org": os.environ.get("QUICKML_ORG", ""),
        "environment": os.environ.get("QUICKML_ENVIRONMENT", "Development"),
        "label": os.environ.get("QUICKML_MODEL_LABEL", "GLM-4.7-Flash"),
        "client_id": os.environ.get("ZOHO_CLIENT_ID", ""),
        "client_secret": os.environ.get("ZOHO_CLIENT_SECRET", ""),
        "refresh_token": os.environ.get("ZOHO_REFRESH_TOKEN", ""),
        "accounts": os.environ.get("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.com"),
    },
}

PUBLIC_URL = os.environ.get("CONDUIT_PUBLIC_URL", "").rstrip("/")
REINDEX_DELAY = float(os.environ.get("REINDEX_DELAY", "120") or 120)
DUP_THRESHOLD = float(os.environ.get("DUP_THRESHOLD", tickets.DEFAULT_THRESHOLD) or tickets.DEFAULT_THRESHOLD)

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# a project may not end in --b: that name is its second index slot (see repos.py)
NAMESPACE_RE = re.compile(r"^(?!.*--b$)[a-z0-9_-]{1,64}$")
TOKEN_RE = re.compile(r"^cdt_(o[a-z0-9]{9})_([a-f0-9]{32})$")
ENDPOINT_RE = re.compile(r"^https://([a-z.]+)/api/v2/applications/\d+/incoming\?appkey=[A-Za-z0-9+/=_%-]+$")
# Cliq data centres; an org's token only works on its own one
CLIQ_HOSTS = ["cliq.zoho.com", "cliq.zoho.in", "cliq.zoho.eu", "cliq.zoho.com.au",
              "cliq.zoho.jp", "cliq.zohocloud.ca", "cliq.zoho.sa", "cliq.zoho.com.cn"]

jobs = {}             # namespace -> job dict
jobs_lock = threading.Lock()
pending = {}          # repo key -> files waiting for the push delay to pass
pending_lock = threading.Lock()
host_cache = {}       # index name -> host


def now():
    return datetime.now(timezone.utc)


def stamp(moment):
    return moment.strftime("%d %b %H:%M UTC")


def sign(tenant):
    return hmac.new(SECRET.encode(), tenant.encode(), hashlib.sha256).hexdigest()[:32]


def tenant_of(given):
    """The tenant a secret belongs to: "" for the original org, None if it is not valid."""
    if not SECRET or not given:
        return None
    if hmac.compare_digest(given, SECRET):
        return ""
    m = TOKEN_RE.match(given)
    if m and hmac.compare_digest(m.group(2), sign(m.group(1))):
        return m.group(1)
    return None


def caller():
    return tenant_of(request.headers.get("X-Conduit-Secret", ""))


@app.before_request
def guard_registry():
    """No project may be called like the registry namespace."""
    ns = request.args.get("namespace", "")
    if not ns and request.is_json:
        ns = str((request.get_json(silent=True) or {}).get("namespace", ""))
    if ns.startswith("__"):
        return jsonify({"error": "namespace must not start with __"}), 400


def scoped(tenant, namespace):
    """Where a project's data lives in Pinecone for this tenant."""
    return f"{tenant}-{namespace}" if tenant else namespace


def get_host():
    if INDEX_NAME not in host_cache:
        host_cache[INDEX_NAME] = index_repo.ensure_index(PINECONE_KEY, INDEX_NAME)
    return host_cache[INDEX_NAME]


def notify(email, text):
    """Message one person through the Cliq bot. Best effort only."""
    if not (CLIQ_TOKEN and email):
        return
    params = {"zapikey": CLIQ_TOKEN}
    if CLIQ_APPKEYS:
        params["appkey"] = CLIQ_APPKEYS[0]
    try:
        requests.post(f"{CLIQ_BASE}/api/v2/bots/{CLIQ_BOT}/message",
                      params=params, json={"text": text, "userids": email}, timeout=20)
    except requests.RequestException:
        pass


def summary(job):
    state = job["state"]
    if state == "running":
        if job["stage"] == "upload" and job["total"]:
            pct = int(100 * job["done"] / job["total"])
            return f"indexing, {pct}% ({job['done']}/{job['total']} chunks)"
        if job["stage"] == "upload":
            return f"indexing, {job['done']} chunks uploaded so far"
        return "indexing, downloading the repo"
    if state == "ready" and job.get("kind") == "update":
        return (f"updated {job['files']} changed files after a push, "
                f"finished {stamp(job['finished'])}")
    if state == "ready":
        return (f"indexed, {job['chunks']} chunks from {job['files']} files, "
                f"finished {stamp(job['finished'])}")
    return f"indexing failed: {job['error']}"


def public(job, namespace):
    return {
        "namespace": namespace,
        "state": job["state"],
        "summary": summary(job),
        "repo": job["repo"],
        "files": job.get("files"),
        "chunks": job.get("chunks"),
        "error": job.get("error"),
    }


def worker(key, namespace, repo, token, ref, email, rkey=None):
    """key is the project's data in Pinecone; namespace is the name the org knows.

    The new index is built in the project's other slot while /route keeps using the
    active one, then the project switches over and the old slot is cleared.
    """
    job = jobs[key]

    def progress(stage, done, total):
        job["stage"], job["done"], job["total"] = stage, done, total

    try:
        record = (repos.load(get_host(), PINECONE_KEY, SECRET, rkey) if rkey else None) or {}
        active = record.get("slot", "a")
        target = repos.other_slot(active)
        files, chunks, host = index_repo.run_index(
            repo, repos.physical(key, target), token, PINECONE_KEY, INDEX_NAME, ref, progress)
        host_cache[INDEX_NAME] = host
        if rkey:
            record = repos.load(host, PINECONE_KEY, SECRET, rkey) or record
            record.update(slot=target, indexed_at=int(time.time()))
            repos.save(host, PINECONE_KEY, SECRET, rkey, record)
            index_repo.clear_namespace(host, PINECONE_KEY, repos.physical(key, active))
        job.update(state="ready", files=files, chunks=chunks, finished=now())
        notify(email, f"✅ Indexing finished for *{namespace}* ({repo}): "
                      f"{chunks} chunks from {files} files. /bug can now search it.")
    except index_repo.IndexerError as err:
        job.update(state="failed", error=str(err), finished=now())
        notify(email, f"⚠️ Indexing failed for *{namespace}* ({repo}): {err}")
    except Exception as err:  # keep the service alive whatever happens
        job.update(state="failed", error=f"unexpected error: {err}", finished=now())
        notify(email, f"⚠️ Indexing failed for *{namespace}* ({repo}) with an unexpected error.")


def ai_label():
    names = []
    if quickml.configured(LLM["quickml"]):
        names.append(LLM["quickml"]["label"])
    if LLM["key"]:
        names.append(LLM["model"])
    return " → ".join(names) if names else "off"


@app.get("/")
def health():
    problems = []
    if not SECRET:
        problems.append("CONDUIT_SECRET is not set")
    if not PINECONE_KEY:
        problems.append("PINECONE_API_KEY is not set")
    return jsonify({
        "service": "conduit-indexer", "ok": not problems, "problems": problems,
        "ai": ai_label(),
        "registration": "open" if (SECRET and REGISTER_KEY) else "off",
        "reminders": f"every {TICK_MINUTES:g} min" if TICK_MINUTES > 0 else "on demand",
    })


@app.post("/index")
def start_index():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    repo = str(body.get("repo", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    token = str(body.get("token", "")).strip()
    ref = str(body.get("ref", "")).strip() or None
    email = str(body.get("notify", "")).strip()
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)

    rkey = repos.repo_key(tenant, namespace)

    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify(public(current, namespace)), 409
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": repo, "started": now()}
    try:
        auto = remember_repo(rkey, tenant, namespace, repo, token, ref)
    except (index_repo.IndexerError, requests.RequestException) as err:
        auto = {"state": "off", "reason": f"could not save the repo settings: {err.__class__.__name__}"}
        rkey = None
    threading.Thread(target=worker, args=(key, namespace, repo, token, ref, email, rkey),
                     daemon=True).start()
    out = public(jobs[key], namespace)
    out["auto_update"] = auto
    return jsonify(out), 202


def public_base():
    if PUBLIC_URL:
        return PUBLIC_URL
    proto = request.headers.get("X-Forwarded-Proto", "https")
    return f"{proto}://{request.host}".replace("http://", "https://")


def remember_repo(rkey, tenant, namespace, repo, token, ref):
    """Store the repo for automatic updates and make sure GitHub has a webhook for it.
    Returns {"state": "on"} or {"state": "off", "reason": ...} for the admin."""
    host = get_host()
    record = repos.load(host, PINECONE_KEY, SECRET, rkey) or {}
    if record.get("hook_id") and record.get("repo") != repo:
        repos.delete_hook(record.get("repo", ""), record.get("token", ""), record.get("hook_id"))
        record.pop("hook_id", None)
    record.update(tenant=tenant, namespace=namespace, repo=repo, ref=ref or "")
    if token:
        record["token"] = token
    auto = {"state": "on"}
    if not record.get("hook_id"):
        record["hook_secret"] = repos.new_hook_secret()
        hook_id, reason = repos.create_hook(repo, record.get("token", ""),
                                            f"{public_base()}/github/{rkey}", record["hook_secret"])
        if hook_id:
            record["hook_id"] = hook_id
        else:
            auto = {"state": "off", "reason": reason}
    repos.save(host, PINECONE_KEY, SECRET, rkey, record)
    return auto


@app.post("/github/<rkey>")
def github_push(rkey):
    """GitHub calls this on every push to a repo Conduit created a webhook for."""
    if not re.fullmatch(r"[a-f0-9]{24}", rkey) or not (PINECONE_KEY and SECRET):
        return jsonify({"error": "unknown hook"}), 404
    try:
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
    except (index_repo.IndexerError, requests.RequestException):
        return jsonify({"error": "could not read the repo settings"}), 502
    if not record:
        return jsonify({"error": "unknown hook"}), 404
    if not repos.signature_ok(record.get("hook_secret", ""), request.get_data(),
                              request.headers.get("X-Hub-Signature-256", "")):
        return jsonify({"error": "bad signature"}), 401
    event = request.headers.get("X-GitHub-Event", "")
    if event == "ping":
        return jsonify({"ok": True})
    if event != "push":
        return jsonify({"ignored": event}), 202
    change = repos.read_push(request.get_json(silent=True) or {}, record.get("ref", ""))
    if change is None:
        return jsonify({"ignored": "not the tracked branch"}), 202
    schedule_update(rkey, change)
    return jsonify({"queued": len(change["changed"]) + len(change["removed"]), "full": change["full"]}), 202


def schedule_update(rkey, change, delay=None):
    """Collect pushes and update the index once they stop for REINDEX_DELAY seconds."""
    with pending_lock:
        waiting = pending.setdefault(rkey, {"full": False, "changed": set(), "removed": set(), "timer": None})
        for path in change["changed"]:
            waiting["changed"].add(path)
            waiting["removed"].discard(path)
        for path in change["removed"]:
            waiting["removed"].add(path)
            waiting["changed"].discard(path)
        waiting["full"] = waiting["full"] or change["full"]
        if waiting["timer"]:
            waiting["timer"].cancel()
        timer = threading.Timer(REINDEX_DELAY if delay is None else delay, apply_update, args=(rkey,))
        timer.daemon = True
        waiting["timer"] = timer
        timer.start()


def apply_update(rkey):
    """Bring one project's index up to date with what was pushed."""
    with pending_lock:
        waiting = pending.pop(rkey, None)
    if not waiting:
        return
    try:
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
    except (index_repo.IndexerError, requests.RequestException):
        record = None
    if not record:
        return
    key = scoped(record["tenant"], record["namespace"])
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            # an index run is busy: try again a little later with the same changes
            change = {"full": waiting["full"], "changed": waiting["changed"], "removed": waiting["removed"]}
            threading.Thread(target=schedule_update, args=(rkey, change, 60), daemon=True).start()
            return
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": record["repo"], "started": now()}
    full = waiting["full"] or not record.get("indexed_at")
    if full:
        worker(key, record["namespace"], record["repo"], record.get("token", ""),
               record.get("ref") or None, "", rkey)
        return
    job = jobs[key]
    job["stage"] = "update"
    try:
        target = repos.physical(key, record.get("slot", "a"))
        updated, gone, chunks = index_repo.update_files(
            get_host(), PINECONE_KEY, target, record["repo"], record.get("token", ""),
            record.get("ref") or None, waiting["changed"], waiting["removed"])
        record.update(updated_at=int(time.time()), updated_files=updated + gone)
        repos.save(get_host(), PINECONE_KEY, SECRET, rkey, record)
        total = index_repo.namespace_count(get_host(), PINECONE_KEY, target)
        job.update(state="ready", kind="update", files=updated + gone, chunks=total or chunks,
                   finished=now())
    except (index_repo.IndexerError, requests.RequestException) as err:
        job.update(state="failed", error=f"update after a push failed: {err}", finished=now())
    except Exception as err:  # keep the service alive whatever happens
        job.update(state="failed", error=f"unexpected error: {err}", finished=now())


@app.post("/forget")
def forget_repo():
    """Delete a project's chunks from Pinecone when it is removed from Conduit."""
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    namespace = str(body.get("namespace", "")).strip()
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify({"error": "this project is being indexed right now"}), 409
        jobs.pop(key, None)
    rkey = repos.repo_key(tenant, namespace)
    with pending_lock:
        waiting = pending.pop(rkey, None)
        if waiting and waiting["timer"]:
            waiting["timer"].cancel()
    try:
        ok = index_repo.clear_namespace(get_host(), PINECONE_KEY, repos.physical(key, "a"))
        ok = index_repo.clear_namespace(get_host(), PINECONE_KEY, repos.physical(key, "b")) and ok
        tickets.forget(get_host(), PINECONE_KEY, key)
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
        if record:
            repos.delete_hook(record.get("repo", ""), record.get("token", ""), record.get("hook_id"))
            repos.remove(get_host(), PINECONE_KEY, rkey)
    except (index_repo.IndexerError, requests.RequestException) as err:
        return jsonify({"error": f"could not reach Pinecone: {err.__class__.__name__}"}), 502
    if not ok:
        return jsonify({"error": "Pinecone refused to delete the data"}), 502
    return jsonify({"namespace": namespace, "deleted": True})


@app.post("/route")
def route_bug():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    repo = str(body.get("repo", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    people = body.get("people") or {}
    if isinstance(people, str):
        try:
            people = json.loads(people)
        except ValueError:
            people = {}
    try:
        search_ns = repos.active_namespace(get_host(), PINECONE_KEY, SECRET, scoped(tenant, namespace),
                                           tenant, namespace)
    except (index_repo.IndexerError, requests.RequestException):
        search_ns = scoped(tenant, namespace)
    req = {
        "repo": repo, "namespace": search_ns, "token": str(body.get("token", "")).strip(),
        "title": str(body.get("title", ""))[:300], "description": str(body.get("description", ""))[:3000],
        "severity": str(body.get("severity", "")), "people": people,
        "ask_questions": str(body.get("ask_questions", "")).lower() in ("true", "1", "yes"),
        "ref": str(body.get("ref", "")).strip(),
    }
    try:
        result = router.route(req, get_host(), PINECONE_KEY, LLM)
    except (router.RouteError, index_repo.IndexerError) as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=str(err))
        return jsonify({"error": str(err)}), 422
    except requests.RequestException as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=f"network error: {err.__class__.__name__}")
        return jsonify({"error": f"network error: {err.__class__.__name__}"}), 502
    log_route(tenant, repo, namespace, req["ref"], people, result=result)
    return jsonify(result)


def log_route(tenant, repo, namespace, ref, people, result=None, error=""):
    """One line per /route call in the Catalyst logs: what was searched, blamed and matched."""
    line = {"event": "route", "tenant": tenant, "repo": repo, "namespace": namespace,
            "ref": ref or "(default branch)", "linked_people": len(people or {})}
    if error:
        line["error"] = error
    if result:
        line["debug"] = result.get("debug", {})
        line["candidates"] = [{"key": c["key"], "cliq": c["cliq"], "score": c["score"],
                               "share": c.get("share"), "confidence": c.get("confidence")}
                              for c in result.get("candidates", [])]
        line["code_confidence"] = result.get("code_confidence")
        line["warnings"] = result.get("warnings", [])
        line["ai"] = result.get("ai_model") or "search order"
    print("CONDUIT " + json.dumps(line, default=str), flush=True)


def ticket_request():
    """Common checks for the /tickets endpoints. Returns (body, error response)."""
    tenant = caller()
    if tenant is None:
        return None, (jsonify({"error": "wrong or missing secret"}), 401)
    if not PINECONE_KEY:
        return None, (jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500)
    body = request.get_json(silent=True) or {}
    namespace = str(body.get("namespace", "")).strip()
    if not NAMESPACE_RE.match(namespace):
        return None, (jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400)
    body["namespace"] = scoped(tenant, namespace)
    return body, None


@app.post("/tickets/add")
def tickets_add():
    body, err = ticket_request()
    if err:
        return err
    record_id = str(body.get("record_id", "")).strip()
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400
    try:
        tickets.add(get_host(), PINECONE_KEY, body["namespace"], record_id,
                    str(body.get("ticket_id", "")), str(body.get("title", ""))[:300],
                    str(body.get("text", ""))[:3000])
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not store the ticket: {e.__class__.__name__}"}), 502
    return jsonify({"stored": record_id})


@app.post("/tickets/remove")
def tickets_remove():
    body, err = ticket_request()
    if err:
        return err
    record_id = str(body.get("record_id", "")).strip()
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400
    try:
        ok = tickets.remove(get_host(), PINECONE_KEY, body["namespace"], record_id)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"removed": record_id, "ok": ok})


@app.post("/tickets/clear")
def tickets_clear():
    """Forget every stored ticket of one project (used by /sweep). The code index is kept."""
    body, err = ticket_request()
    if err:
        return err
    try:
        ok = tickets.forget(get_host(), PINECONE_KEY, body["namespace"])
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"cleared": bool(ok)})


@app.post("/tickets/similar")
def tickets_similar():
    body, err = ticket_request()
    if err:
        return err
    info = {}
    try:
        matches = tickets.similar(get_host(), PINECONE_KEY, body["namespace"],
                                  str(body.get("title", ""))[:300], str(body.get("text", ""))[:3000],
                                  DUP_THRESHOLD, body.get("exclude"), info)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not search tickets: {e.__class__.__name__}", "matches": []}), 502
    # "best" is the closest score seen, even below the threshold; handy for tuning DUP_THRESHOLD
    return jsonify({"matches": matches, "threshold": DUP_THRESHOLD, "best": info.get("best", 0)})


def ping_org(endpoint, token):
    """Tick one org's copy of Conduit through its extension incoming webhook."""
    try:
        r = requests.post(endpoint, params={"zapikey": token}, json={"action": "tick"}, timeout=30)
        return r.status_code
    except requests.RequestException:
        return "error"


def ping_orgs():
    """Tick every org that turned on reminders. Returns (orgs, delivered, error)."""
    if not (PINECONE_KEY and SECRET):
        return 0, 0, "the indexer has no PINECONE_API_KEY"
    try:
        orgs = registry.load(get_host(), PINECONE_KEY, SECRET)
    except (index_repo.IndexerError, requests.RequestException) as err:
        return 0, 0, f"could not read the org list: {err}"
    ok = 0
    for org in orgs:
        if ping_org(org["endpoint"], org["token"]) in (200, 204):
            ok += 1
    return len(orgs), ok, None


def ticker():
    """Ping the bots every TICK_MINUTES while the service is awake."""
    while True:
        time.sleep(TICK_MINUTES * 60)
        ping_orgs()


if SECRET and TICK_MINUTES > 0:
    threading.Thread(target=ticker, daemon=True).start()


@app.get("/flux.png")
def flux_icon():
    """Flux's avatar, public so Cliq can show it on messages Conduit posts in channels."""
    resp = send_from_directory(os.path.dirname(os.path.abspath(__file__)), "flux.png", mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.post("/register")
def register():
    """A new org installed Conduit: hand it a secret of its own."""
    given = request.headers.get("X-Conduit-Register", "")
    if not (SECRET and REGISTER_KEY) or not hmac.compare_digest(given, REGISTER_KEY):
        return jsonify({"error": "registration is not open"}), 401
    tenant = "o" + "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(9))
    return jsonify({"secret": f"cdt_{tenant}_{sign(tenant)}"})


@app.get("/privacy")
def privacy_page():
    """Public privacy policy, linked from the Zoho Marketplace listing."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal"), "privacy.html", mimetype="text/html")


@app.get("/terms")
def terms_page():
    """Public terms of service, linked from the Zoho Marketplace listing."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal"), "terms.html", mimetype="text/html")


@app.route("/tick", methods=["GET", "POST"])
def tick():
    """For an outside scheduler: keeps the service awake and sends reminders now."""
    given = request.args.get("secret", "") or request.headers.get("X-Conduit-Secret", "")
    if not SECRET or not hmac.compare_digest(given, SECRET):
        return jsonify({"error": "wrong or missing secret"}), 401
    orgs, delivered, error = ping_orgs()
    out = {"orgs": orgs, "delivered": delivered}
    if error:
        out["error"] = error
    return jsonify(out)


@app.post("/reminders/connect")
def reminders_connect():
    """Store an org's extension token, after checking it really reaches their Conduit."""
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    endpoint = str(body.get("endpoint", "")).strip()
    token = str(body.get("token", "")).strip()
    m = ENDPOINT_RE.match(endpoint)
    if not m or m.group(1) not in CLIQ_HOSTS:
        return jsonify({"error": "endpoint must be the extension's incoming webhook URL"}), 400
    given_host = m.group(1)
    if not token or len(token) > 500:
        return jsonify({"error": "token is missing"}), 400
    tried = {}
    # the org's own data centre first, then the others
    for host in [given_host] + [h for h in CLIQ_HOSTS if h != given_host]:
        candidate = endpoint.replace(given_host, host, 1)
        code = ping_org(candidate, token)
        tried[host] = code
        if code in (200, 204):
            try:
                registry.save(get_host(), PINECONE_KEY, SECRET, tenant, candidate, token)
            except (index_repo.IndexerError, requests.RequestException) as e:
                return jsonify({"error": f"could not save: {e.__class__.__name__}"}), 502
            return jsonify({"connected": True, "host": host})
    return jsonify({"error": "Cliq did not accept the token", "tried": tried}), 422


@app.post("/reminders/disconnect")
def reminders_disconnect():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    try:
        ok = registry.remove(get_host(), PINECONE_KEY, tenant)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"disconnected": ok})


@app.get("/status")
def status():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    namespace = request.args.get("namespace", "").strip()
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)
    record = None
    if PINECONE_KEY:
        try:
            record = repos.load(get_host(), PINECONE_KEY, SECRET, repos.repo_key(tenant, namespace))
        except (index_repo.IndexerError, requests.RequestException):
            record = None
    auto = {"state": "on" if (record and record.get("hook_id")) else "off",
            "updated_at": (record or {}).get("updated_at"), "updated_files": (record or {}).get("updated_files")}
    job = jobs.get(key)
    if job:
        out = public(job, namespace)
        out["auto_update"] = auto
        return jsonify(out)
    # The service may have restarted since the last run; ask Pinecone instead.
    count = None
    if PINECONE_KEY:
        try:
            count = index_repo.namespace_count(get_host(), PINECONE_KEY,
                                               repos.physical(key, (record or {}).get("slot", "a")))
        except index_repo.IndexerError:
            count = None
    if count:
        text = f"indexed, {count} chunks"
        state = "ready"
    elif count == 0:
        text, state = "not indexed yet", "missing"
    else:
        text, state = "status unknown", "unknown"
    return jsonify({"namespace": namespace, "state": state, "summary": text, "chunks": count,
                    "auto_update": auto})


if __name__ == "__main__":
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT") or os.environ.get("PORT") or 9000)
    app.run(host="0.0.0.0", port=port, threaded=True)
// ---- router.py ----
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
// ---- app.py ----
"""
Conduit indexer service.

A small web service that does Conduit's heavy lifting:
    POST /index   index a GitHub repo into Pinecone (Add repo, Re-index)
    GET  /status  indexing progress for the /repo card
    POST /route   decide which code a bug is about and who should own it (/bug)
    POST /tickets/add, /tickets/remove, /tickets/similar, /tickets/clear
                  open tickets as searchable text, to catch duplicate reports
    POST /register
                  give a newly installed org its own secret
    POST /github/<key>
                  GitHub push webhook: re-index only the files a push changed
    POST /reminders/connect, /reminders/disconnect
                  an org hands over (or withdraws) an extension token so the
                  indexer can tick its copy of Conduit for reminders
Each org's secret carries a tenant id, and every namespace it touches is stored
under that tenant, so two orgs with a project called "payments" never mix.
It also "ticks" every org that turned on reminders, so each one's copy of
Conduit can send them.

Environment variables
    CONDUIT_SECRET      required. Any long random string. Signs every org's secret.
                        Used directly as a secret it acts as the original org,
                        whose data has no tenant prefix.
    CONDUIT_REGISTER_KEY optional. Turns on /register; the same value is built into
                        the extension, so only Conduit itself can sign orgs up.
    PINECONE_API_KEY    required. Your Pinecone API key.
    PINECONE_INDEX      optional. Index name, default "conduit".
    CLIQ_WEBHOOK_TOKEN  optional. A Cliq webhook token; when set, the service
                        messages the person through the bot when indexing ends.
    CLIQ_BOT_NAME       optional. Bot unique name, default "conduit".
    CLIQ_APPKEYS        App key of the Conduit extension (My Extensions -> Conduit ->
                        Connectors), used with CLIQ_WEBHOOK_TOKEN for the
                        "indexing finished" message.
    CLIQ_API_BASE       optional. Default https://cliq.zoho.com; India accounts
                        use https://cliq.zoho.in
    TICK_MINUTES        optional. How often to tick connected orgs while awake,
                        default 10. 0 turns it off (Catalyst Cron calls /tick anyway).
    LLM_API_KEY         optional. Turns on AI triage. A Groq key (gsk_...) by default.
    LLM_BASE_URL        optional. Any OpenAI-compatible endpoint; default Groq,
                        https://api.groq.com/openai/v1
    LLM_MODEL           optional. Default openai/gpt-oss-120b
    QUICKML_URL, QUICKML_ENDPOINT_KEY, QUICKML_ORG
                        optional. A Catalyst QuickML LLM endpoint; when set (with the
                        three ZOHO_* values) it is tried first and LLM_* is the fallback.
    QUICKML_ENVIRONMENT optional. Development (default) or Production
    QUICKML_MODEL_LABEL optional. Name shown in /repo, default GLM-4.7-Flash
    ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN
                        a Zoho Self Client with scope QuickML.deployment.READ
    ZOHO_ACCOUNTS_URL   optional. Default https://accounts.zoho.com
    CONDUIT_PUBLIC_URL  optional. This service's public https address, used for the
                        GitHub webhooks it creates; by default taken from the request.
    REINDEX_DELAY       optional. Seconds to wait after the last push before updating
                        the index, default 120, so a burst of pushes updates it once.
    ROUTE_BUDGET        optional. Seconds /route may take in total, default 20, so /bug
                        answers before Cliq gives up; the AI is skipped first.
    DUP_THRESHOLD       optional. How similar a new bug must be to an open ticket
                        to count as a duplicate, default 0.87

Runs on Zoho Catalyst AppSail (reads X_ZOHO_CATALYST_LISTEN_PORT) or any host
that sets PORT, such as Render.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request, send_from_directory

import index_repo
import quickml
import registry
import repos
import router
import tickets

app = Flask(__name__)

SECRET = os.environ.get("CONDUIT_SECRET", "")
REGISTER_KEY = os.environ.get("CONDUIT_REGISTER_KEY", "")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "conduit")
CLIQ_TOKEN = os.environ.get("CLIQ_WEBHOOK_TOKEN", "")
CLIQ_BOT = os.environ.get("CLIQ_BOT_NAME", "conduit")
CLIQ_BASE = os.environ.get("CLIQ_API_BASE", "https://cliq.zoho.com").rstrip("/")
CLIQ_APPKEYS = [k.strip() for k in os.environ.get("CLIQ_APPKEYS", "").split(",") if k.strip()]
TICK_MINUTES = float(os.environ.get("TICK_MINUTES", "10") or 0)
LLM = {
    "key": os.environ.get("LLM_API_KEY", ""),
    "base": os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    "model": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
    # Catalyst QuickML (Zoho-hosted model); tried first when configured
    "quickml": {
        "url": os.environ.get("QUICKML_URL", ""),
        "endpoint_key": os.environ.get("QUICKML_ENDPOINT_KEY", ""),
        "org": os.environ.get("QUICKML_ORG", ""),
        "environment": os.environ.get("QUICKML_ENVIRONMENT", "Development"),
        "label": os.environ.get("QUICKML_MODEL_LABEL", "GLM-4.7-Flash"),
        "client_id": os.environ.get("ZOHO_CLIENT_ID", ""),
        "client_secret": os.environ.get("ZOHO_CLIENT_SECRET", ""),
        "refresh_token": os.environ.get("ZOHO_REFRESH_TOKEN", ""),
        "accounts": os.environ.get("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.com"),
    },
}

PUBLIC_URL = os.environ.get("CONDUIT_PUBLIC_URL", "").rstrip("/")
REINDEX_DELAY = float(os.environ.get("REINDEX_DELAY", "120") or 120)
DUP_THRESHOLD = float(os.environ.get("DUP_THRESHOLD", tickets.DEFAULT_THRESHOLD) or tickets.DEFAULT_THRESHOLD)

REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# a project may not end in --b: that name is its second index slot (see repos.py)
NAMESPACE_RE = re.compile(r"^(?!.*--b$)[a-z0-9_-]{1,64}$")
TOKEN_RE = re.compile(r"^cdt_(o[a-z0-9]{9})_([a-f0-9]{32})$")
ENDPOINT_RE = re.compile(r"^https://([a-z.]+)/api/v2/applications/\d+/incoming\?appkey=[A-Za-z0-9+/=_%-]+$")
# Cliq data centres; an org's token only works on its own one
CLIQ_HOSTS = ["cliq.zoho.com", "cliq.zoho.in", "cliq.zoho.eu", "cliq.zoho.com.au",
              "cliq.zoho.jp", "cliq.zohocloud.ca", "cliq.zoho.sa", "cliq.zoho.com.cn"]

jobs = {}             # namespace -> job dict
jobs_lock = threading.Lock()
pending = {}          # repo key -> files waiting for the push delay to pass
pending_lock = threading.Lock()
host_cache = {}       # index name -> host


def now():
    return datetime.now(timezone.utc)


def stamp(moment):
    return moment.strftime("%d %b %H:%M UTC")


def sign(tenant):
    return hmac.new(SECRET.encode(), tenant.encode(), hashlib.sha256).hexdigest()[:32]


def tenant_of(given):
    """The tenant a secret belongs to: "" for the original org, None if it is not valid."""
    if not SECRET or not given:
        return None
    if hmac.compare_digest(given, SECRET):
        return ""
    m = TOKEN_RE.match(given)
    if m and hmac.compare_digest(m.group(2), sign(m.group(1))):
        return m.group(1)
    return None


def caller():
    return tenant_of(request.headers.get("X-Conduit-Secret", ""))


@app.before_request
def guard_registry():
    """No project may be called like the registry namespace."""
    ns = request.args.get("namespace", "")
    if not ns and request.is_json:
        ns = str((request.get_json(silent=True) or {}).get("namespace", ""))
    if ns.startswith("__"):
        return jsonify({"error": "namespace must not start with __"}), 400


def scoped(tenant, namespace):
    """Where a project's data lives in Pinecone for this tenant."""
    return f"{tenant}-{namespace}" if tenant else namespace


def get_host():
    if INDEX_NAME not in host_cache:
        host_cache[INDEX_NAME] = index_repo.ensure_index(PINECONE_KEY, INDEX_NAME)
    return host_cache[INDEX_NAME]


def notify(email, text):
    """Message one person through the Cliq bot. Best effort only."""
    if not (CLIQ_TOKEN and email):
        return
    params = {"zapikey": CLIQ_TOKEN}
    if CLIQ_APPKEYS:
        params["appkey"] = CLIQ_APPKEYS[0]
    try:
        requests.post(f"{CLIQ_BASE}/api/v2/bots/{CLIQ_BOT}/message",
                      params=params, json={"text": text, "userids": email}, timeout=20)
    except requests.RequestException:
        pass


def summary(job):
    state = job["state"]
    if state == "running":
        if job["stage"] == "upload" and job["total"]:
            pct = int(100 * job["done"] / job["total"])
            return f"indexing, {pct}% ({job['done']}/{job['total']} chunks)"
        if job["stage"] == "upload":
            return f"indexing, {job['done']} chunks uploaded so far"
        return "indexing, downloading the repo"
    if state == "ready" and job.get("kind") == "update":
        return (f"updated {job['files']} changed files after a push, "
                f"finished {stamp(job['finished'])}")
    if state == "ready":
        return (f"indexed, {job['chunks']} chunks from {job['files']} files, "
                f"finished {stamp(job['finished'])}")
    return f"indexing failed: {job['error']}"


def public(job, namespace):
    return {
        "namespace": namespace,
        "state": job["state"],
        "summary": summary(job),
        "repo": job["repo"],
        "files": job.get("files"),
        "chunks": job.get("chunks"),
        "error": job.get("error"),
    }


def worker(key, namespace, repo, token, ref, email, rkey=None):
    """key is the project's data in Pinecone; namespace is the name the org knows.

    The new index is built in the project's other slot while /route keeps using the
    active one, then the project switches over and the old slot is cleared.
    """
    job = jobs[key]

    def progress(stage, done, total):
        job["stage"], job["done"], job["total"] = stage, done, total

    try:
        record = (repos.load(get_host(), PINECONE_KEY, SECRET, rkey) if rkey else None) or {}
        active = record.get("slot", "a")
        target = repos.other_slot(active)
        files, chunks, host = index_repo.run_index(
            repo, repos.physical(key, target), token, PINECONE_KEY, INDEX_NAME, ref, progress)
        host_cache[INDEX_NAME] = host
        if rkey:
            record = repos.load(host, PINECONE_KEY, SECRET, rkey) or record
            record.update(slot=target, indexed_at=int(time.time()))
            repos.save(host, PINECONE_KEY, SECRET, rkey, record)
            index_repo.clear_namespace(host, PINECONE_KEY, repos.physical(key, active))
        job.update(state="ready", files=files, chunks=chunks, finished=now())
        notify(email, f"✅ Indexing finished for *{namespace}* ({repo}): "
                      f"{chunks} chunks from {files} files. /bug can now search it.")
    except index_repo.IndexerError as err:
        job.update(state="failed", error=str(err), finished=now())
        notify(email, f"⚠️ Indexing failed for *{namespace}* ({repo}): {err}")
    except Exception as err:  # keep the service alive whatever happens
        job.update(state="failed", error=f"unexpected error: {err}", finished=now())
        notify(email, f"⚠️ Indexing failed for *{namespace}* ({repo}) with an unexpected error.")


def ai_label():
    names = []
    if quickml.configured(LLM["quickml"]):
        names.append(LLM["quickml"]["label"])
    if LLM["key"]:
        names.append(LLM["model"])
    return " → ".join(names) if names else "off"


@app.get("/")
def health():
    problems = []
    if not SECRET:
        problems.append("CONDUIT_SECRET is not set")
    if not PINECONE_KEY:
        problems.append("PINECONE_API_KEY is not set")
    return jsonify({
        "service": "conduit-indexer", "ok": not problems, "problems": problems,
        "ai": ai_label(),
        "registration": "open" if (SECRET and REGISTER_KEY) else "off",
        "reminders": f"every {TICK_MINUTES:g} min" if TICK_MINUTES > 0 else "on demand",
    })


@app.post("/index")
def start_index():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    repo = str(body.get("repo", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    token = str(body.get("token", "")).strip()
    ref = str(body.get("ref", "")).strip() or None
    email = str(body.get("notify", "")).strip()
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)

    rkey = repos.repo_key(tenant, namespace)

    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify(public(current, namespace)), 409
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": repo, "started": now()}
    try:
        auto = remember_repo(rkey, tenant, namespace, repo, token, ref)
    except (index_repo.IndexerError, requests.RequestException) as err:
        auto = {"state": "off", "reason": f"could not save the repo settings: {err.__class__.__name__}"}
        rkey = None
    threading.Thread(target=worker, args=(key, namespace, repo, token, ref, email, rkey),
                     daemon=True).start()
    out = public(jobs[key], namespace)
    out["auto_update"] = auto
    return jsonify(out), 202


def public_base():
    if PUBLIC_URL:
        return PUBLIC_URL
    proto = request.headers.get("X-Forwarded-Proto", "https")
    return f"{proto}://{request.host}".replace("http://", "https://")


def remember_repo(rkey, tenant, namespace, repo, token, ref):
    """Store the repo for automatic updates and make sure GitHub has a webhook for it.
    Returns {"state": "on"} or {"state": "off", "reason": ...} for the admin."""
    host = get_host()
    record = repos.load(host, PINECONE_KEY, SECRET, rkey) or {}
    if record.get("hook_id") and record.get("repo") != repo:
        repos.delete_hook(record.get("repo", ""), record.get("token", ""), record.get("hook_id"))
        record.pop("hook_id", None)
    record.update(tenant=tenant, namespace=namespace, repo=repo, ref=ref or "")
    if token:
        record["token"] = token
    auto = {"state": "on"}
    if not record.get("hook_id"):
        record["hook_secret"] = repos.new_hook_secret()
        hook_id, reason = repos.create_hook(repo, record.get("token", ""),
                                            f"{public_base()}/github/{rkey}", record["hook_secret"])
        if hook_id:
            record["hook_id"] = hook_id
        else:
            auto = {"state": "off", "reason": reason}
    repos.save(host, PINECONE_KEY, SECRET, rkey, record)
    return auto


@app.post("/github/<rkey>")
def github_push(rkey):
    """GitHub calls this on every push to a repo Conduit created a webhook for."""
    if not re.fullmatch(r"[a-f0-9]{24}", rkey) or not (PINECONE_KEY and SECRET):
        return jsonify({"error": "unknown hook"}), 404
    try:
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
    except (index_repo.IndexerError, requests.RequestException):
        return jsonify({"error": "could not read the repo settings"}), 502
    if not record:
        return jsonify({"error": "unknown hook"}), 404
    if not repos.signature_ok(record.get("hook_secret", ""), request.get_data(),
                              request.headers.get("X-Hub-Signature-256", "")):
        return jsonify({"error": "bad signature"}), 401
    event = request.headers.get("X-GitHub-Event", "")
    if event == "ping":
        return jsonify({"ok": True})
    if event != "push":
        return jsonify({"ignored": event}), 202
    change = repos.read_push(request.get_json(silent=True) or {}, record.get("ref", ""))
    if change is None:
        return jsonify({"ignored": "not the tracked branch"}), 202
    schedule_update(rkey, change)
    return jsonify({"queued": len(change["changed"]) + len(change["removed"]), "full": change["full"]}), 202


def schedule_update(rkey, change, delay=None):
    """Collect pushes and update the index once they stop for REINDEX_DELAY seconds."""
    with pending_lock:
        waiting = pending.setdefault(rkey, {"full": False, "changed": set(), "removed": set(), "timer": None})
        for path in change["changed"]:
            waiting["changed"].add(path)
            waiting["removed"].discard(path)
        for path in change["removed"]:
            waiting["removed"].add(path)
            waiting["changed"].discard(path)
        waiting["full"] = waiting["full"] or change["full"]
        if waiting["timer"]:
            waiting["timer"].cancel()
        timer = threading.Timer(REINDEX_DELAY if delay is None else delay, apply_update, args=(rkey,))
        timer.daemon = True
        waiting["timer"] = timer
        timer.start()


def apply_update(rkey):
    """Bring one project's index up to date with what was pushed."""
    with pending_lock:
        waiting = pending.pop(rkey, None)
    if not waiting:
        return
    try:
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
    except (index_repo.IndexerError, requests.RequestException):
        record = None
    if not record:
        return
    key = scoped(record["tenant"], record["namespace"])
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            # an index run is busy: try again a little later with the same changes
            change = {"full": waiting["full"], "changed": waiting["changed"], "removed": waiting["removed"]}
            threading.Thread(target=schedule_update, args=(rkey, change, 60), daemon=True).start()
            return
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": record["repo"], "started": now()}
    full = waiting["full"] or not record.get("indexed_at")
    if full:
        worker(key, record["namespace"], record["repo"], record.get("token", ""),
               record.get("ref") or None, "", rkey)
        return
    job = jobs[key]
    job["stage"] = "update"
    try:
        target = repos.physical(key, record.get("slot", "a"))
        updated, gone, chunks = index_repo.update_files(
            get_host(), PINECONE_KEY, target, record["repo"], record.get("token", ""),
            record.get("ref") or None, waiting["changed"], waiting["removed"])
        record.update(updated_at=int(time.time()), updated_files=updated + gone)
        repos.save(get_host(), PINECONE_KEY, SECRET, rkey, record)
        total = index_repo.namespace_count(get_host(), PINECONE_KEY, target)
        job.update(state="ready", kind="update", files=updated + gone, chunks=total or chunks,
                   finished=now())
    except (index_repo.IndexerError, requests.RequestException) as err:
        job.update(state="failed", error=f"update after a push failed: {err}", finished=now())
    except Exception as err:  # keep the service alive whatever happens
        job.update(state="failed", error=f"unexpected error: {err}", finished=now())


@app.post("/forget")
def forget_repo():
    """Delete a project's chunks from Pinecone when it is removed from Conduit."""
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    namespace = str(body.get("namespace", "")).strip()
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify({"error": "this project is being indexed right now"}), 409
        jobs.pop(key, None)
    rkey = repos.repo_key(tenant, namespace)
    with pending_lock:
        waiting = pending.pop(rkey, None)
        if waiting and waiting["timer"]:
            waiting["timer"].cancel()
    try:
        ok = index_repo.clear_namespace(get_host(), PINECONE_KEY, repos.physical(key, "a"))
        ok = index_repo.clear_namespace(get_host(), PINECONE_KEY, repos.physical(key, "b")) and ok
        tickets.forget(get_host(), PINECONE_KEY, key)
        record = repos.load(get_host(), PINECONE_KEY, SECRET, rkey)
        if record:
            repos.delete_hook(record.get("repo", ""), record.get("token", ""), record.get("hook_id"))
            repos.remove(get_host(), PINECONE_KEY, rkey)
    except (index_repo.IndexerError, requests.RequestException) as err:
        return jsonify({"error": f"could not reach Pinecone: {err.__class__.__name__}"}), 502
    if not ok:
        return jsonify({"error": "Pinecone refused to delete the data"}), 502
    return jsonify({"namespace": namespace, "deleted": True})


@app.post("/route")
def route_bug():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    repo = str(body.get("repo", "")).strip()
    namespace = str(body.get("namespace", "")).strip()
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    people = body.get("people") or {}
    if isinstance(people, str):
        try:
            people = json.loads(people)
        except ValueError:
            people = {}
    try:
        search_ns = repos.active_namespace(get_host(), PINECONE_KEY, SECRET, scoped(tenant, namespace),
                                           tenant, namespace)
    except (index_repo.IndexerError, requests.RequestException):
        search_ns = scoped(tenant, namespace)
    req = {
        "repo": repo, "namespace": search_ns, "token": str(body.get("token", "")).strip(),
        "title": str(body.get("title", ""))[:300], "description": str(body.get("description", ""))[:3000],
        "severity": str(body.get("severity", "")), "people": people,
        "ask_questions": str(body.get("ask_questions", "")).lower() in ("true", "1", "yes"),
        "ref": str(body.get("ref", "")).strip(),
    }
    try:
        result = router.route(req, get_host(), PINECONE_KEY, LLM)
    except (router.RouteError, index_repo.IndexerError) as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=str(err))
        return jsonify({"error": str(err)}), 422
    except requests.RequestException as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=f"network error: {err.__class__.__name__}")
        return jsonify({"error": f"network error: {err.__class__.__name__}"}), 502
    log_route(tenant, repo, namespace, req["ref"], people, result=result)
    return jsonify(result)


def log_route(tenant, repo, namespace, ref, people, result=None, error=""):
    """One line per /route call in the Catalyst logs: what was searched, blamed and matched."""
    line = {"event": "route", "tenant": tenant, "repo": repo, "namespace": namespace,
            "ref": ref or "(default branch)", "linked_people": len(people or {})}
    if error:
        line["error"] = error
    if result:
        line["debug"] = result.get("debug", {})
        line["candidates"] = [{"key": c["key"], "cliq": c["cliq"], "score": c["score"],
                               "share": c.get("share"), "confidence": c.get("confidence")}
                              for c in result.get("candidates", [])]
        line["code_confidence"] = result.get("code_confidence")
        line["warnings"] = result.get("warnings", [])
        line["ai"] = result.get("ai_model") or "search order"
    print("CONDUIT " + json.dumps(line, default=str), flush=True)


def ticket_request():
    """Common checks for the /tickets endpoints. Returns (body, error response)."""
    tenant = caller()
    if tenant is None:
        return None, (jsonify({"error": "wrong or missing secret"}), 401)
    if not PINECONE_KEY:
        return None, (jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500)
    body = request.get_json(silent=True) or {}
    namespace = str(body.get("namespace", "")).strip()
    if not NAMESPACE_RE.match(namespace):
        return None, (jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400)
    body["namespace"] = scoped(tenant, namespace)
    return body, None


@app.post("/tickets/add")
def tickets_add():
    body, err = ticket_request()
    if err:
        return err
    record_id = str(body.get("record_id", "")).strip()
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400
    try:
        tickets.add(get_host(), PINECONE_KEY, body["namespace"], record_id,
                    str(body.get("ticket_id", "")), str(body.get("title", ""))[:300],
                    str(body.get("text", ""))[:3000])
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not store the ticket: {e.__class__.__name__}"}), 502
    return jsonify({"stored": record_id})


@app.post("/tickets/remove")
def tickets_remove():
    body, err = ticket_request()
    if err:
        return err
    record_id = str(body.get("record_id", "")).strip()
    if not record_id:
        return jsonify({"error": "record_id is required"}), 400
    try:
        ok = tickets.remove(get_host(), PINECONE_KEY, body["namespace"], record_id)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"removed": record_id, "ok": ok})


@app.post("/tickets/clear")
def tickets_clear():
    """Forget every stored ticket of one project (used by /sweep). The code index is kept."""
    body, err = ticket_request()
    if err:
        return err
    try:
        ok = tickets.forget(get_host(), PINECONE_KEY, body["namespace"])
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"cleared": bool(ok)})


@app.post("/tickets/similar")
def tickets_similar():
    body, err = ticket_request()
    if err:
        return err
    info = {}
    try:
        matches = tickets.similar(get_host(), PINECONE_KEY, body["namespace"],
                                  str(body.get("title", ""))[:300], str(body.get("text", ""))[:3000],
                                  DUP_THRESHOLD, body.get("exclude"), info)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not search tickets: {e.__class__.__name__}", "matches": []}), 502
    # "best" is the closest score seen, even below the threshold; handy for tuning DUP_THRESHOLD
    return jsonify({"matches": matches, "threshold": DUP_THRESHOLD, "best": info.get("best", 0)})


def ping_org(endpoint, token):
    """Tick one org's copy of Conduit through its extension incoming webhook."""
    try:
        r = requests.post(endpoint, params={"zapikey": token}, json={"action": "tick"}, timeout=30)
        return r.status_code
    except requests.RequestException:
        return "error"


def ping_orgs():
    """Tick every org that turned on reminders. Returns (orgs, delivered, error)."""
    if not (PINECONE_KEY and SECRET):
        return 0, 0, "the indexer has no PINECONE_API_KEY"
    try:
        orgs = registry.load(get_host(), PINECONE_KEY, SECRET)
    except (index_repo.IndexerError, requests.RequestException) as err:
        return 0, 0, f"could not read the org list: {err}"
    ok = 0
    for org in orgs:
        if ping_org(org["endpoint"], org["token"]) in (200, 204):
            ok += 1
    return len(orgs), ok, None


def ticker():
    """Ping the bots every TICK_MINUTES while the service is awake."""
    while True:
        time.sleep(TICK_MINUTES * 60)
        ping_orgs()


if SECRET and TICK_MINUTES > 0:
    threading.Thread(target=ticker, daemon=True).start()


@app.get("/flux.png")
def flux_icon():
    """Flux's avatar, public so Cliq can show it on messages Conduit posts in channels."""
    resp = send_from_directory(os.path.dirname(os.path.abspath(__file__)), "flux.png", mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.post("/register")
def register():
    """A new org installed Conduit: hand it a secret of its own."""
    given = request.headers.get("X-Conduit-Register", "")
    if not (SECRET and REGISTER_KEY) or not hmac.compare_digest(given, REGISTER_KEY):
        return jsonify({"error": "registration is not open"}), 401
    tenant = "o" + "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(9))
    return jsonify({"secret": f"cdt_{tenant}_{sign(tenant)}"})


@app.get("/privacy")
def privacy_page():
    """Public privacy policy, linked from the Zoho Marketplace listing."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal"), "privacy.html", mimetype="text/html")


@app.get("/terms")
def terms_page():
    """Public terms of service, linked from the Zoho Marketplace listing."""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal"), "terms.html", mimetype="text/html")


@app.route("/tick", methods=["GET", "POST"])
def tick():
    """For an outside scheduler: keeps the service awake and sends reminders now."""
    given = request.args.get("secret", "") or request.headers.get("X-Conduit-Secret", "")
    if not SECRET or not hmac.compare_digest(given, SECRET):
        return jsonify({"error": "wrong or missing secret"}), 401
    orgs, delivered, error = ping_orgs()
    out = {"orgs": orgs, "delivered": delivered}
    if error:
        out["error"] = error
    return jsonify(out)


@app.post("/reminders/connect")
def reminders_connect():
    """Store an org's extension token, after checking it really reaches their Conduit."""
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    body = request.get_json(silent=True) or {}
    endpoint = str(body.get("endpoint", "")).strip()
    token = str(body.get("token", "")).strip()
    m = ENDPOINT_RE.match(endpoint)
    if not m or m.group(1) not in CLIQ_HOSTS:
        return jsonify({"error": "endpoint must be the extension's incoming webhook URL"}), 400
    given_host = m.group(1)
    if not token or len(token) > 500:
        return jsonify({"error": "token is missing"}), 400
    tried = {}
    # the org's own data centre first, then the others
    for host in [given_host] + [h for h in CLIQ_HOSTS if h != given_host]:
        candidate = endpoint.replace(given_host, host, 1)
        code = ping_org(candidate, token)
        tried[host] = code
        if code in (200, 204):
            try:
                registry.save(get_host(), PINECONE_KEY, SECRET, tenant, candidate, token)
            except (index_repo.IndexerError, requests.RequestException) as e:
                return jsonify({"error": f"could not save: {e.__class__.__name__}"}), 502
            return jsonify({"connected": True, "host": host})
    return jsonify({"error": "Cliq did not accept the token", "tried": tried}), 422


@app.post("/reminders/disconnect")
def reminders_disconnect():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    try:
        ok = registry.remove(get_host(), PINECONE_KEY, tenant)
    except (index_repo.IndexerError, requests.RequestException) as e:
        return jsonify({"error": f"could not reach Pinecone: {e.__class__.__name__}"}), 502
    return jsonify({"disconnected": ok})


@app.get("/status")
def status():
    tenant = caller()
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    namespace = request.args.get("namespace", "").strip()
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    key = scoped(tenant, namespace)
    record = None
    if PINECONE_KEY:
        try:
            record = repos.load(get_host(), PINECONE_KEY, SECRET, repos.repo_key(tenant, namespace))
        except (index_repo.IndexerError, requests.RequestException):
            record = None
    auto = {"state": "on" if (record and record.get("hook_id")) else "off",
            "updated_at": (record or {}).get("updated_at"), "updated_files": (record or {}).get("updated_files")}
    job = jobs.get(key)
    if job:
        out = public(job, namespace)
        out["auto_update"] = auto
        return jsonify(out)
    # The service may have restarted since the last run; ask Pinecone instead.
    count = None
    if PINECONE_KEY:
        try:
            count = index_repo.namespace_count(get_host(), PINECONE_KEY,
                                               repos.physical(key, (record or {}).get("slot", "a")))
        except index_repo.IndexerError:
            count = None
    if count:
        text = f"indexed, {count} chunks"
        state = "ready"
    elif count == 0:
        text, state = "not indexed yet", "missing"
    else:
        text, state = "status unknown", "unknown"
    return jsonify({"namespace": namespace, "state": state, "summary": text, "chunks": count,
                    "auto_update": auto})


if __name__ == "__main__":
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT") or os.environ.get("PORT") or 9000)
    app.run(host="0.0.0.0", port=port, threaded=True)
