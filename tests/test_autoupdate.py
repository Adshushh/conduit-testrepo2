"""Auto re-index on push: repo settings, GitHub webhooks, the push delay, index slots."""
import hashlib
import hmac
import json
import time
import unittest
from unittest import mock

import support
from support import Response

import index_repo
import repos
from test_app import AppTest


def push(ref="refs/heads/main", commits=None, forced=False, deleted=False, default_branch="main"):
    return {"ref": ref, "forced": forced, "deleted": deleted, "after": "abc123def4567890",
            "repository": {"default_branch": default_branch}, "commits": commits or []}


def commit(added=(), modified=(), removed=()):
    return {"added": list(added), "modified": list(modified), "removed": list(removed)}


class ReadPushTest(support.TestCase):
    def test_collects_changed_and_removed_files(self):
        change = repos.read_push(push(commits=[commit(added=["a.py"], modified=["b.py"]),
                                               commit(removed=["c.py"])]), "main")
        self.assertEqual(change["changed"], {"a.py", "b.py"})
        self.assertEqual(change["removed"], {"c.py"})
        self.assertFalse(change["full"])

    def test_later_commits_win(self):
        change = repos.read_push(push(commits=[commit(removed=["a.py"]), commit(added=["a.py"]),
                                               commit(modified=["b.py"]), commit(removed=["b.py"])]), "main")
        self.assertEqual(change["changed"], {"a.py"})
        self.assertEqual(change["removed"], {"b.py"})

    def test_other_branches_and_deleted_branches_are_ignored(self):
        self.assertIsNone(repos.read_push(push(ref="refs/heads/feature"), "main"))
        self.assertIsNone(repos.read_push(push(deleted=True), "main"))

    def test_default_branch_when_none_was_chosen(self):
        self.assertIsNotNone(repos.read_push(push(ref="refs/heads/trunk", default_branch="trunk"), ""))
        self.assertIsNone(repos.read_push(push(ref="refs/heads/main", default_branch="trunk"), ""))

    def test_full_rebuild_cases(self):
        self.assertTrue(repos.read_push(push(forced=True, commits=[commit(modified=["a.py"])]), "main")["full"])
        many_commits = [commit(modified=[f"f{i}.py"]) for i in range(repos.MAX_PUSH_COMMITS)]
        self.assertTrue(repos.read_push(push(commits=many_commits), "main")["full"])
        many_files = [commit(modified=[f"f{i}.py" for i in range(repos.MAX_CHANGED_FILES + 1)])]
        self.assertTrue(repos.read_push(push(commits=many_files), "main")["full"])


class SignatureTest(support.TestCase):
    def test_valid_and_invalid(self):
        body = b'{"ref":"refs/heads/main"}'
        good = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        self.assertTrue(repos.signature_ok("s3cret", body, good))
        self.assertFalse(repos.signature_ok("other", body, good))
        self.assertFalse(repos.signature_ok("s3cret", body + b" ", good))
        self.assertFalse(repos.signature_ok("s3cret", body, ""))
        self.assertFalse(repos.signature_ok("", body, good))


class CreateHookTest(support.TestCase):
    def create(self, status, data=None):
        with mock.patch.object(repos.requests, "post", return_value=Response(status, data or {})) as post:
            return repos.create_hook("acme/shop", "tok", "https://idx/github/k", "sec"), post

    def test_created(self):
        (hook_id, reason), post = self.create(201, {"id": 5})
        self.assertEqual((hook_id, reason), (5, ""))
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["events"], ["push"])
        self.assertEqual(sent["config"]["secret"], "sec")

    def test_token_without_webhook_permission(self):
        (hook_id, reason), _ = self.create(403)
        self.assertIsNone(hook_id)
        self.assertIn("Webhooks: Read and write", reason)


class FileUpdateTest(support.TestCase):
    """index_repo: finding, fetching and replacing one file's chunks."""

    def test_delete_file_chunks_uses_the_file_prefix(self):
        prefix = index_repo.file_id("src/a.py") + "-"
        listing = [Response(200, {"vectors": [{"id": prefix + "0"}], "pagination": {"next": "t"}}),
                   Response(200, {"vectors": [{"id": prefix + "1"}]})]
        with mock.patch.object(index_repo.requests, "get", side_effect=listing) as get, \
                mock.patch.object(index_repo.requests, "post", return_value=Response(200, {})) as post:
            self.assertEqual(index_repo.delete_file_chunks("host", "key", "ns", "src/a.py"), 2)
        self.assertEqual(get.call_args_list[0].kwargs["params"]["prefix"], prefix)
        self.assertEqual(post.call_args.kwargs["json"], {"ids": [prefix + "0", prefix + "1"], "namespace": "ns"})

    def test_fetch_file(self):
        ok = Response(200, {})
        ok.content = b"print('hi')\n"
        binary = Response(200, {})
        binary.content = b"\x89PNG\x00\x00"
        with mock.patch.object(index_repo.requests, "get", side_effect=[ok, binary, Response(404, {})]):
            self.assertEqual(index_repo.fetch_file("acme/shop", "tok", "a.py", "main"), "print('hi')\n")
            self.assertIsNone(index_repo.fetch_file("acme/shop", "tok", "logo.py"))
            self.assertIsNone(index_repo.fetch_file("acme/shop", "tok", "gone.py"))

    def test_update_files(self):
        texts = {"src/a.py": "x = 1\n" * 50, "src/gone.py": None}
        with mock.patch.object(index_repo, "delete_file_chunks", return_value=1) as delete, \
                mock.patch.object(index_repo, "fetch_file", side_effect=lambda r, t, p, ref: texts[p]) as fetch, \
                mock.patch.object(index_repo, "upsert") as upsert:
            result = index_repo.update_files("host", "key", "ns", "acme/shop", "tok", "main",
                                             {"src/a.py", "src/gone.py", "logo.png"}, {"old.py"})
        self.assertEqual(result, (1, 2, 2))      # a.py updated; old.py and gone.py removed; 2 chunks
        deleted = sorted(c.args[3] for c in delete.call_args_list)
        self.assertEqual(deleted, ["logo.png", "old.py", "src/a.py", "src/gone.py"])
        fetched = sorted(c.args[2] for c in fetch.call_args_list)
        self.assertEqual(fetched, ["src/a.py", "src/gone.py"])  # unsupported files are not fetched
        self.assertEqual(upsert.call_args.args[2], "ns")


class IndexSlotsTest(AppTest):
    """/index keeps the repo, sets up the webhook and builds the other slot."""

    def index(self, run_index=None, **body):
        body = dict({"repo": "acme/shop", "namespace": "shop", "token": "tok"}, **body)
        run_index = run_index or mock.Mock(return_value=(3, 12, "fake-index.pinecone.io"))
        with mock.patch.object(self.app.index_repo, "run_index", run_index):
            resp = self.client.post("/index", json=body, headers=self.headers())
            key = f"{self.tenant()}-shop"
            for _ in range(100):
                if self.app.jobs[key]["state"] != "running":
                    break
                time.sleep(0.01)
        return resp, run_index

    def record(self):
        return self.store[repos.repo_key(self.tenant(), "shop")]

    def test_webhook_is_created_and_settings_kept(self):
        resp, _ = self.index(ref="develop")
        self.assertEqual(resp.get_json()["auto_update"], {"state": "on"})
        url = self.fake["create_hook"].call_args.args[2]
        self.assertTrue(url.endswith("/github/" + repos.repo_key(self.tenant(), "shop")))
        self.assertTrue(url.startswith("https://"))
        record = self.record()
        self.assertEqual((record["repo"], record["ref"], record["token"], record["hook_id"]),
                         ("acme/shop", "develop", "tok", 77))

    def test_webhook_not_created_twice(self):
        self.index()
        self.index()
        self.assertEqual(self.fake["create_hook"].call_count, 1)

    def test_without_webhook_permission(self):
        self.fake["create_hook"].return_value = (None, "the token needs the Webhooks: Read and write permission")
        resp, run_index = self.index()
        self.assertEqual(resp.get_json()["auto_update"]["state"], "off")
        self.assertIn("Webhooks", resp.get_json()["auto_update"]["reason"])
        run_index.assert_called_once()           # indexing still happens

    def test_builds_the_other_slot_then_switches(self):
        key = f"{self.tenant()}-shop"
        _, run_index = self.index()
        self.assertEqual(run_index.call_args.args[1], key + "--b")
        self.assertEqual(self.record()["slot"], "b")
        self.assertEqual(self.fake["clear_namespace"].call_args.args[2], key)       # old slot emptied
        _, run_index = self.index()
        self.assertEqual(run_index.call_args.args[1], key)
        self.assertEqual(self.record()["slot"], "a")

    def test_failed_build_keeps_the_active_slot(self):
        self.index()
        failing = mock.Mock(side_effect=index_repo.IndexerError("GitHub rejected the token."))
        self.index(run_index=failing)
        self.assertEqual(self.record()["slot"], "b")

    def test_route_searches_the_active_slot(self):
        self.index()
        with mock.patch.object(self.app.router, "route", return_value={"candidates": []}) as route, \
                mock.patch("builtins.print"):
            self.client.post("/route", json={"repo": "acme/shop", "namespace": "shop"}, headers=self.headers())
        self.assertEqual(route.call_args.args[0]["namespace"], f"{self.tenant()}-shop--b")

    def test_slot_names_cannot_be_projects(self):
        resp = self.client.post("/index", json={"repo": "acme/shop", "namespace": "shop--b"}, headers=self.headers())
        self.assertEqual(resp.status_code, 400)


class PushWebhookTest(AppTest):
    def setUp(self):
        super().setUp()
        self.rkey = repos.repo_key(self.tenant(), "shop")
        self.store[self.rkey] = {"tenant": self.tenant(), "namespace": "shop", "repo": "acme/shop",
                                 "ref": "main", "token": "tok", "hook_id": 77, "hook_secret": "s3cret",
                                 "slot": "b", "indexed_at": 1}
        self.app.pending.clear()
        self.addCleanup(self.clear_pending)

    def clear_pending(self):
        for waiting in self.app.pending.values():
            if waiting["timer"]:
                waiting["timer"].cancel()
        self.app.pending.clear()

    def deliver(self, payload, event="push", secret="s3cret", rkey=None):
        body = json.dumps(payload).encode()
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return self.client.post(f"/github/{rkey or self.rkey}", data=body, content_type="application/json",
                                headers={"X-GitHub-Event": event, "X-Hub-Signature-256": sig})

    def test_unknown_hook_and_bad_signature(self):
        self.assertEqual(self.deliver(push(), rkey="0" * 24).status_code, 404)
        self.assertEqual(self.deliver(push(), rkey="not-a-key").status_code, 404)
        self.assertEqual(self.deliver(push(), secret="wrong").status_code, 401)

    def test_ping(self):
        self.assertEqual(self.deliver({"zen": "hi"}, event="ping").get_json(), {"ok": True})

    def test_other_branch_is_ignored(self):
        resp = self.deliver(push(ref="refs/heads/feature", commits=[commit(modified=["a.py"])]))
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(self.app.pending, {})

    def test_pushes_are_collected_and_applied_once(self):
        self.deliver(push(commits=[commit(modified=["a.py"]), commit(removed=["b.py"])]))
        self.deliver(push(commits=[commit(added=["b.py"]), commit(modified=["c.py"])]))
        waiting = self.app.pending[self.rkey]
        self.assertEqual((waiting["changed"], waiting["removed"]), ({"a.py", "b.py", "c.py"}, set()))
        with mock.patch.object(self.app.index_repo, "update_files", return_value=(3, 0, 9)) as update, \
                mock.patch.object(self.app.index_repo, "namespace_count", return_value=120):
            self.app.apply_update(self.rkey)
        update.assert_called_once()
        args = update.call_args.args
        self.assertEqual(args[2], f"{self.tenant()}-shop--b")       # the active slot
        self.assertEqual((args[3], args[4], args[5]), ("acme/shop", "tok", "main"))
        self.assertEqual((args[6], args[7]), ({"a.py", "b.py", "c.py"}, set()))
        status = self.client.get("/status?namespace=shop", headers=self.headers()).get_json()
        self.assertEqual(status["state"], "ready")
        self.assertIn("updated 3 changed files", status["summary"])
        self.assertEqual(status["chunks"], 120)
        self.assertEqual(self.record()["updated_files"], 3)

    def record(self):
        return self.store[self.rkey]

    def test_the_wait_restarts_on_every_push(self):
        self.deliver(push(commits=[commit(modified=["a.py"])]))
        first = self.app.pending[self.rkey]["timer"]
        self.deliver(push(commits=[commit(modified=["b.py"])]))
        self.assertTrue(first.finished.is_set())                # the earlier wait was cancelled
        self.assertFalse(self.app.pending[self.rkey]["timer"].finished.is_set())

    def test_force_push_rebuilds_in_full(self):
        self.deliver(push(forced=True, commits=[commit(modified=["a.py"])]))
        with mock.patch.object(self.app.index_repo, "run_index", return_value=(5, 50, "fake-index.pinecone.io")) as run_index, \
                mock.patch.object(self.app.index_repo, "update_files") as update:
            self.app.apply_update(self.rkey)
        update.assert_not_called()
        self.assertEqual(run_index.call_args.args[1], f"{self.tenant()}-shop")   # built in the other slot
        self.assertEqual(self.record()["slot"], "a")

    def test_waits_while_an_index_run_is_busy(self):
        key = f"{self.tenant()}-shop"
        self.app.jobs[key] = {"state": "running", "stage": "upload", "done": 0, "total": 0, "repo": "acme/shop"}
        self.deliver(push(commits=[commit(modified=["a.py"])]))
        with mock.patch.object(self.app, "schedule_update") as again, \
                mock.patch.object(self.app.index_repo, "update_files") as update:
            self.app.apply_update(self.rkey)
            for _ in range(100):
                if again.called:
                    break
                time.sleep(0.01)
        update.assert_not_called()
        self.assertEqual(again.call_args.args[1]["changed"], {"a.py"})

    def test_failed_update_is_reported(self):
        self.deliver(push(commits=[commit(modified=["a.py"])]))
        with mock.patch.object(self.app.index_repo, "update_files",
                               side_effect=index_repo.IndexerError("GitHub rejected the token.")):
            self.app.apply_update(self.rkey)
        status = self.client.get("/status?namespace=shop", headers=self.headers()).get_json()
        self.assertEqual(status["state"], "failed")
        self.assertIn("GitHub rejected the token", status["error"])

    def test_forget_cancels_waiting_updates(self):
        self.deliver(push(commits=[commit(modified=["a.py"])]))
        with mock.patch.object(self.app.tickets, "forget", return_value=True):
            self.client.post("/forget", json={"namespace": "shop"}, headers=self.headers())
        self.assertNotIn(self.rkey, self.app.pending)


if __name__ == "__main__":
    unittest.main()
