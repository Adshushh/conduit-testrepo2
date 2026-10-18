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




# a project may not end in --b: that name is its second index slot (see repos.py)
# Cliq data centres; an org's token only works on its own one

























def worker(key, namespace, repo, token, ref, email, rkey=None):
    """key is the project's data in Pinecone; namespace is the name the org knows.

    The new index is built in the project's other slot while /route keeps using the
    active one, then the project switches over and the old slot is cleared.
    """


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






@app.post("/index")
def start_index():


    return jsonify(out), 202










def apply_update(rkey):
    """Bring one project's index up to date with what was pushed."""
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
    return jsonify({"namespace": namespace, "deleted": True})


@app.post("/route")
def route_bug():
    return jsonify(result)










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












@app.get("/flux.png")
def flux_icon():
    """Flux's avatar, public so Cliq can show it on messages Conduit posts in channels."""
    resp = send_from_directory(os.path.dirname(os.path.abspath(__file__)), "flux.png", mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp














@app.get("/status")
def status():
    if tenant is None:
        return jsonify({"error": "wrong or missing secret"}), 401
    if not NAMESPACE_RE.match(namespace):
        return jsonify({"error": "namespace must be lowercase letters, digits, - or _"}), 400
    record = None
    if PINECONE_KEY:
        try:
            record = repos.load(get_host(), PINECONE_KEY, SECRET, repos.repo_key(tenant, namespace))
        except (index_repo.IndexerError, requests.RequestException):
            record = None
    if job:
        out = public(job, namespace)
        out["auto_update"] = auto
        return jsonify(out)
    # The service may have restarted since the last run; ask Pinecone instead.
    if count:
        text = f"indexed, {count} chunks"
        state = "ready"
    elif count == 0:
        text, state = "not indexed yet", "missing"
    else:
        text, state = "status unknown", "unknown"


