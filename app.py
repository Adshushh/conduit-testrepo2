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

REGISTER_KEY = os.environ.get("CONDUIT_REGISTER_KEY", "")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "conduit")
CLIQ_BOT = os.environ.get("CLIQ_BOT_NAME", "conduit")
CLIQ_APPKEYS = [k.strip() for k in os.environ.get("CLIQ_APPKEYS", "").split(",") if k.strip()]
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


# a project may not end in --b: that name is its second index slot (see repos.py)
# Cliq data centres; an org's token only works on its own one







def sign(tenant):
    return hmac.new(SECRET.encode(), tenant.encode(), hashlib.sha256).hexdigest()[:32]




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




def get_host():
    if INDEX_NAME not in host_cache:
        host_cache[INDEX_NAME] = index_repo.ensure_index(PINECONE_KEY, INDEX_NAME)
    return host_cache[INDEX_NAME]




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
    ref = str(body.get("ref", "")).strip() or None
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    key = scoped(tenant, namespace)


    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify(public(current, namespace)), 409
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": repo, "started": now()}
    out = public(jobs[key], namespace)
    out["auto_update"] = auto
    return jsonify(out), 202




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
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            # an index run is busy: try again a little later with the same changes
            change = {"full": waiting["full"], "changed": waiting["changed"], "removed": waiting["removed"]}
            threading.Thread(target=schedule_update, args=(rkey, change, 60), daemon=True).start()
            return
        jobs[key] = {"state": "running", "stage": "download", "done": 0, "total": 0,
                     "repo": record["repo"], "started": now()}
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
    if not PINECONE_KEY:
        return jsonify({"error": "the indexer has no PINECONE_API_KEY"}), 500
    namespace = str(body.get("namespace", "")).strip()
    key = scoped(tenant, namespace)
    with jobs_lock:
        current = jobs.get(key)
        if current and current["state"] == "running":
            return jsonify({"error": "this project is being indexed right now"}), 409
        jobs.pop(key, None)
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
    return jsonify({"namespace": namespace, "deleted": True})


@app.post("/route")
def route_bug():
    if not REPO_RE.match(repo):
        return jsonify({"error": "repo must look like owner/name"}), 400
    people = body.get("people") or {}
    try:
        search_ns = repos.active_namespace(get_host(), PINECONE_KEY, SECRET, scoped(tenant, namespace),
                                           tenant, namespace)
    except (index_repo.IndexerError, requests.RequestException):
        search_ns = scoped(tenant, namespace)
    try:
        result = router.route(req, get_host(), PINECONE_KEY, LLM)
    except (router.RouteError, index_repo.IndexerError) as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=str(err))
        return jsonify({"error": str(err)}), 422
    except requests.RequestException as err:
        log_route(tenant, repo, namespace, req["ref"], people, error=f"network error: {err.__class__.__name__}")
        return jsonify({"error": f"network error: {err.__class__.__name__}"}), 502
    return jsonify(result)




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


