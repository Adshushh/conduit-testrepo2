#!/usr/bin/env python3
"""
Conduit repo indexer.

Downloads a GitHub repo, splits its source files into chunks and stores them
in a Pinecone index that embeds text for you (integrated embedding). Conduit's
/bug command searches this index to find the files a bug most likely lives in.

The hosted service (app.py) imports this file and calls run_index(). You can
also run it by hand:

    pip install requests
    export GITHUB_TOKEN=github_pat_xxx
    export PINECONE_API_KEY=pcsk_xxx

    python index_repo.py --repo fastapi/full-stack-fastapi-template --namespace fastapi-template
    python index_repo.py --namespace fastapi-template --query "password reset email never arrives"
    python index_repo.py --repo owner/name --namespace x --dry-run
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time

import requests


# chunks collected before they are uploaded; keeps memory flat on big repos

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".java", ".kt", ".go",
    ".rb", ".php", ".cs", ".c", ".h", ".cpp", ".hpp", ".rs", ".swift", ".dart",
    ".html", ".css", ".scss", ".vue", ".svelte", ".sql", ".sh", ".yml", ".yaml",
    ".toml", ".md", ".dg", ".prisma", ".graphql",
}
SKIP_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"}


class IndexerError(Exception):
    """A problem the person can fix: bad repo name, token, key and so on."""


def pinecone_headers(api_key, content_type="application/json"):
    return {
        "Api-Key": api_key,
        "X-Pinecone-Api-Version": PINECONE_VERSION,
        "Content-Type": content_type,
        "Accept": "application/json",
    }


def ensure_index(api_key, index_name):
    """Return the index host, creating the index if it does not exist yet."""
    headers = pinecone_headers(api_key)
    resp = requests.get(f"{PINECONE_API}/indexes/{index_name}", headers=headers, timeout=30)
    if resp.status_code == 404:
        body = {
            "name": index_name,
            "cloud": "aws",
            "region": "us-east-1",
            "embed": {"model": EMBED_MODEL, "field_map": {"text": TEXT_FIELD}},
        }
        resp = requests.post(f"{PINECONE_API}/indexes/create-for-model",
                             headers=headers, json=body, timeout=60)
    if resp.status_code == 401:
        raise IndexerError("Pinecone rejected the API key.")
    if resp.status_code not in (200, 201):
        raise IndexerError(f"Pinecone error {resp.status_code}: {resp.text[:200]}")

    info = resp.json()
    for _ in range(100):  # wait up to ~5 minutes for a new index
        if info.get("status", {}).get("ready"):
            return info["host"]
        time.sleep(3)
        info = requests.get(f"{PINECONE_API}/indexes/{index_name}",
                            headers=headers, timeout=30).json()
    raise IndexerError("The Pinecone index did not become ready in time.")


def download_repo(repo, token, ref=None):
    """Yield (path, text) for every indexable file in the repo.

    The archive is saved to a temporary file in pieces and read back one file at a
    time, so a large repo never has to fit in memory.
    """
    archive_file.seek(0)

    with archive_file, tarfile.open(fileobj=archive_file, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.size > MAX_FILE_BYTES:
                continue
            # the archive wraps everything in one top-level folder; drop it
            parts = member.name.split("/")[1:]
            if not parts or should_skip(parts):
                continue
            raw = archive.extractfile(member).read()
            if b"\x00" in raw:
                continue  # binary file
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if text.strip():
                yield "/".join(parts), text




def file_id(path):
    """Every chunk id of a file starts with this, so one file's chunks can be found and replaced."""
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def chunk_file(path, text):
    """Split one file into overlapping line-based chunks."""
    lines = text.splitlines()
    step = CHUNK_LINES - CHUNK_OVERLAP
    file_id_ = file_id(path)
    start = 0
    number = 0
    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        body = "\n".join(lines[start:end]).strip()
        if body:
            # the path goes into the text so file and folder names help matching
            chunk_text = f"File: {path}\n{body}"[:CHUNK_MAX_CHARS]
            yield {
                "_id": f"{file_id_}-{number}",
                TEXT_FIELD: chunk_text,
                "path": path,
                "start_line": start + 1,
                "end_line": end,
                "lines": f"{start + 1}-{end}",
            }
            number += 1
        if end == len(lines):
            break
        start += step




def upsert(host, api_key, namespace, records, progress=None):
    url = f"https://{host}/records/namespaces/{namespace}/upsert"
    headers = pinecone_headers(api_key, "application/x-ndjson")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        payload = "\n".join(json.dumps(r) for r in batch).encode("utf-8")
        for attempt in range(6):
            resp = requests.post(url, headers=headers, data=payload, timeout=120)
            if resp.status_code in (200, 201):
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            raise IndexerError(f"Pinecone upload failed {resp.status_code}: {resp.text[:200]}")
        else:
            raise IndexerError("Pinecone kept rejecting the upload; try again in a few minutes.")
        if progress:
            progress(min(i + BATCH_SIZE, len(records)), len(records))




def fetch_file(repo, token, path, ref=None):
    """One file's text from GitHub, or None when it is gone, binary, too big or not text."""
    headers = {"Accept": "application/vnd.github.raw+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}", headers=headers,
                        params={"ref": ref} if ref else None, timeout=60)
    if resp.status_code == 404:
        return None
    if resp.status_code == 401:
        raise IndexerError("GitHub rejected the token.")
    if resp.status_code != 200:
        raise IndexerError(f"GitHub error {resp.status_code} for {path}.")
    raw = resp.content
    if len(raw) > MAX_FILE_BYTES or b"\x00" in raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if text.strip() else None


def update_files(host, api_key, namespace, repo, token, ref, changed, removed):
    """Re-index only the files a push touched. Returns (files updated, files removed, chunks)."""
    updated = gone = chunks = 0
    for path in sorted(set(removed) - set(changed)):
        delete_file_chunks(host, api_key, namespace, path)
        gone += 1
    for path in sorted(set(changed)):
        parts = path.split("/")
        delete_file_chunks(host, api_key, namespace, path)
        if should_skip(parts):
            continue
        text = fetch_file(repo, token, path, ref)
        if text is None:
            gone += 1
            continue
        records = list(chunk_file(path, text))
        if records:
            upsert(host, api_key, namespace, records)
            chunks += len(records)
        updated += 1
    return updated, gone, chunks


def namespace_count(host, api_key, namespace):
    """Chunks currently stored in a namespace, or None if Pinecone won't say."""
    try:
        resp = requests.post(f"https://{host}/describe_index_stats",
                             headers=pinecone_headers(api_key), json={}, timeout=30)
        if resp.status_code != 200:
            return None
        spaces = resp.json().get("namespaces", {})
        info = spaces.get(namespace)
        if info is None:
            return 0
        for key in ("vectorCount", "vector_count", "recordCount", "record_count"):
            if key in info:
                return int(info[key])
    except (requests.RequestException, ValueError):
        return None
    return None




def build_records(repo, token, ref=None):
    records = []
    file_count = 0
    for path, text in download_repo(repo, token, ref):
        file_count += 1
        records.extend(chunk_file(path, text))
    return file_count, records


def run_index(repo, namespace, token, api_key, index_name="conduit", ref=None, progress=None):
    """Index one repo into one namespace. Returns (files, chunks, host).

    Chunks are uploaded in batches while the archive is read, so memory stays flat
    however big the repo is. progress(stage, done, total) reports "download",
    "upload" (total is 0 until the end, as the size is not known up front) and "done".
    """
    say = progress or (lambda *a: None)
    say("download", 0, 0)
    host = ensure_index(api_key, index_name)
    clear_namespace(host, api_key, namespace)
    file_count, uploaded, waiting = 0, 0, []
    for path, text in download_repo(repo, token, ref):
        file_count += 1
        waiting.extend(chunk_file(path, text))
        if len(waiting) >= UPLOAD_EVERY:
            upsert(host, api_key, namespace, waiting)
            uploaded += len(waiting)
            waiting = []
            say("upload", uploaded, 0)
    if waiting:
        upsert(host, api_key, namespace, waiting)
        uploaded += len(waiting)
    if not uploaded:
        raise IndexerError("Nothing to index: the repo is empty or has no supported file types.")
    say("done", uploaded, uploaded)
    return file_count, uploaded, host


def main():

    if not api_key and not args.dry_run:
        sys.exit("Set PINECONE_API_KEY first.")

    try:

        if not args.repo:
            sys.exit("--repo is required when indexing.")

        if args.dry_run:
            file_count, records = build_records(args.repo, os.environ.get("GITHUB_TOKEN"), args.ref)
            print(f"{file_count} files -> {len(records)} chunks")
            for record in records[:5]:
                print(f"  {record['_id']}  {record['path']}  lines {record['lines']}")
            print("Dry run only; nothing was uploaded.")
            return

        def show(stage, done, total):
            if stage == "download":
                print(f"Downloading {args.repo} ...")
            elif stage == "upload" and total:
                print(f"  uploaded {done}/{total} chunks")

        print(f"Index host: {host}")
    except IndexerError as err:
        sys.exit(str(err))


