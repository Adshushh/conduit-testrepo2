"""HTTP endpoints: secrets, org separation, input checks and error replies."""
import time
import unittest
from unittest import mock

import support

app_module = support.load_app()


class AppTest(support.TestCase):
    def setUp(self):
        super().setUp()
        self.app = app_module
        self.client = app_module.app.test_client()
        app_module.jobs.clear()
        self.addCleanup(app_module.jobs.clear)
        # the repo registry lives in Pinecone and webhooks on GitHub: keep both in memory
        self.store = {}
        repos = app_module.repos
        fakes = [
            mock.patch.object(repos, "load", side_effect=lambda h, k, sec, key: dict(self.store[key]) if key in self.store else None),
            mock.patch.object(repos, "save", side_effect=lambda h, k, sec, key, data: self.store.__setitem__(key, dict(data))),
            mock.patch.object(repos, "remove", side_effect=lambda h, k, key: self.store.pop(key, None)),
            mock.patch.object(repos, "create_hook", return_value=(77, "")),
            mock.patch.object(repos, "delete_hook"),
            mock.patch.object(app_module.index_repo, "clear_namespace", return_value=True),
        ]
        self.fake = {}
        for f in fakes:
            self.fake[f.attribute] = f.start()
            self.addCleanup(f.stop)
        self.org_secret = self.register()

    def register(self):
        resp = self.client.post("/register", headers={"X-Conduit-Register": support.REGISTER_KEY})
        self.assertEqual(resp.status_code, 200)
        return resp.get_json()["secret"]

    def headers(self, secret=None):
        return {"X-Conduit-Secret": self.org_secret if secret is None else secret}

    def tenant(self, secret=None):
        return self.app.tenant_of(secret or self.org_secret)


class HealthTest(AppTest):
    def test_health(self):
        body = self.client.get("/").get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["registration"], "open")
        self.assertEqual(body["ai"], "off")

    def test_public_pages(self):
        for path in ("/privacy", "/terms", "/flux.png"):
            resp = self.client.get(path)
            resp.close()
            self.assertEqual(resp.status_code, 200, path)


class RegisterTest(AppTest):
    def test_wrong_register_key(self):
        resp = self.client.post("/register", headers={"X-Conduit-Register": "nope"})
        self.assertEqual(resp.status_code, 401)

    def test_each_org_gets_its_own_valid_secret(self):
        other = self.register()
        self.assertNotEqual(other, self.org_secret)
        self.assertRegex(self.org_secret, r"^cdt_o[a-z0-9]{9}_[a-f0-9]{32}$")
        self.assertNotEqual(self.tenant(), self.tenant(other))

    def test_tenant_of(self):
        self.assertEqual(self.app.tenant_of(support.SECRET), "")          # the original org
        self.assertIsNone(self.app.tenant_of(""))
        self.assertIsNone(self.app.tenant_of("cdt_oabcdefghi_" + "0" * 32))  # forged signature
        self.assertIsNone(self.app.tenant_of(self.org_secret[:-1] + "x"))


class SecretRequiredTest(AppTest):
    def test_every_private_endpoint_needs_a_secret(self):
        calls = [("post", "/index"), ("post", "/forget"), ("post", "/route"), ("post", "/tickets/add"),
                 ("post", "/tickets/remove"), ("post", "/tickets/clear"), ("post", "/tickets/similar"),
                 ("post", "/reminders/connect"), ("post", "/reminders/disconnect"),
                 ("get", "/status?namespace=shop"), ("get", "/tick")]
        for method, path in calls:
            for secret in ("", "wrong"):
                resp = getattr(self.client, method)(path, json={"namespace": "shop"}, headers=self.headers(secret))
                self.assertEqual(resp.status_code, 401, f"{method} {path} with {secret!r}")

    def test_tick_needs_the_main_secret_not_an_org_secret(self):
        self.assertEqual(self.client.get("/tick", headers=self.headers()).status_code, 401)

    def test_registry_namespace_is_off_limits(self):
        resp = self.client.post("/tickets/clear", json={"namespace": "__registry"}, headers=self.headers())
        self.assertEqual(resp.status_code, 400)
        resp = self.client.get("/status?namespace=__registry", headers=self.headers())
        self.assertEqual(resp.status_code, 400)


class RouteEndpointTest(AppTest):
    body = {"repo": "acme/shop", "namespace": "shop", "token": "tok", "title": "Login fails",
            "description": "after submit", "severity": "high", "ask_questions": "true",
            "people": '{"Adshushh": "adesh@acme.com"}'}

    def post(self, body=None, result=None, error=None):
        route = mock.patch.object(self.app.router, "route", return_value=result or {"candidates": []},
                                  side_effect=error)
        with route as fake, mock.patch("builtins.print"):
            resp = self.client.post("/route", json=body or self.body, headers=self.headers())
        return resp, fake

    def test_request_is_scoped_to_the_org(self):
        resp, fake = self.post()
        self.assertEqual(resp.status_code, 200)
        req = fake.call_args.args[0]
        self.assertEqual(req["namespace"], f"{self.tenant()}-shop")
        self.assertEqual(req["people"], {"Adshushh": "adesh@acme.com"})
        self.assertTrue(req["ask_questions"])

    def test_two_orgs_never_share_a_namespace(self):
        _, fake = self.post()
        mine = fake.call_args.args[0]["namespace"]
        self.org_secret = self.register()
        _, fake = self.post()
        self.assertNotEqual(fake.call_args.args[0]["namespace"], mine)

    def test_bad_input(self):
        for change in ({"repo": "not a repo"}, {"namespace": "Bad Name"}, {"namespace": ""}):
            resp, fake = self.post(dict(self.body, **change))
            self.assertEqual(resp.status_code, 400, change)
            fake.assert_not_called()

    def test_routing_error_is_422(self):
        resp, _ = self.post(error=self.app.router.RouteError("no matching code"))
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.get_json()["error"], "no matching code")

    def test_network_error_is_502(self):
        resp, _ = self.post(error=self.app.requests.ConnectionError())
        self.assertEqual(resp.status_code, 502)

    def test_log_line_never_contains_the_token(self):
        with mock.patch.object(self.app.router, "route", return_value={"candidates": [], "debug": {}}), \
                mock.patch("builtins.print") as printed:
            self.client.post("/route", json=self.body, headers=self.headers())
        line = printed.call_args.args[0]
        self.assertTrue(line.startswith("CONDUIT "))
        self.assertNotIn("tok\"", line)


class TicketEndpointTest(AppTest):
    def test_similar_uses_the_org_namespace_and_threshold(self):
        with mock.patch.object(self.app.tickets, "similar", return_value=[]) as similar:
            resp = self.client.post("/tickets/similar", headers=self.headers(),
                                    json={"namespace": "shop", "title": "t", "text": "x", "exclude": "r1"})
        self.assertEqual(resp.status_code, 200)
        args = similar.call_args.args
        self.assertEqual(args[2], f"{self.tenant()}-shop")
        self.assertEqual(args[5], self.app.DUP_THRESHOLD)
        self.assertEqual(args[6], "r1")

    def test_add_needs_a_record_id(self):
        resp = self.client.post("/tickets/add", json={"namespace": "shop"}, headers=self.headers())
        self.assertEqual(resp.status_code, 400)

    def test_clear_forgets_only_tickets(self):
        with mock.patch.object(self.app.tickets, "forget", return_value=True) as forget, \
                mock.patch.object(self.app.index_repo, "clear_namespace") as clear_code:
            resp = self.client.post("/tickets/clear", json={"namespace": "shop"}, headers=self.headers())
        self.assertEqual(resp.get_json(), {"cleared": True})
        forget.assert_called_once()
        clear_code.assert_not_called()


class IndexEndpointTest(AppTest):
    def test_index_runs_in_the_background(self):
        with mock.patch.object(self.app.index_repo, "run_index", return_value=(3, 12, "fake-index.pinecone.io")):
            resp = self.client.post("/index", headers=self.headers(),
                                    json={"repo": "acme/shop", "namespace": "shop", "ref": "develop"})
            self.assertEqual(resp.status_code, 202)
            key = f"{self.tenant()}-shop"
            for _ in range(50):
                if self.app.jobs[key]["state"] != "running":
                    break
                time.sleep(0.02)
        status = self.client.get("/status?namespace=shop", headers=self.headers()).get_json()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["chunks"], 12)
        self.assertEqual(status["auto_update"]["state"], "on")

    def test_second_index_while_running_is_refused(self):
        self.app.jobs[f"{self.tenant()}-shop"] = {"state": "running", "stage": "download", "done": 0,
                                                  "total": 0, "repo": "acme/shop"}
        resp = self.client.post("/index", json={"repo": "acme/shop", "namespace": "shop"}, headers=self.headers())
        self.assertEqual(resp.status_code, 409)
        resp = self.client.post("/forget", json={"namespace": "shop"}, headers=self.headers())
        self.assertEqual(resp.status_code, 409)

    def test_failed_index_is_reported(self):
        error = self.app.index_repo.IndexerError("GitHub rejected the token.")
        with mock.patch.object(self.app.index_repo, "run_index", side_effect=error):
            self.client.post("/index", json={"repo": "acme/shop", "namespace": "shop"}, headers=self.headers())
            key = f"{self.tenant()}-shop"
            for _ in range(50):
                if self.app.jobs[key]["state"] != "running":
                    break
                time.sleep(0.02)
        status = self.client.get("/status?namespace=shop", headers=self.headers()).get_json()
        self.assertEqual(status["state"], "failed")
        self.assertIn("GitHub rejected the token", status["error"])

    def test_status_after_restart_asks_pinecone(self):
        with mock.patch.object(self.app.index_repo, "namespace_count", return_value=0):
            status = self.client.get("/status?namespace=shop", headers=self.headers()).get_json()
        self.assertEqual(status["state"], "missing")

    def test_forget_deletes_code_tickets_webhook_and_settings(self):
        rkey = self.app.repos.repo_key(self.tenant(), "shop")
        self.store[rkey] = {"repo": "acme/shop", "token": "tok", "hook_id": 77, "slot": "b"}
        with mock.patch.object(self.app.tickets, "forget", return_value=True) as forget:
            resp = self.client.post("/forget", json={"namespace": "shop"}, headers=self.headers())
        self.assertEqual(resp.get_json(), {"namespace": "shop", "deleted": True})
        cleared = [c.args[2] for c in self.fake["clear_namespace"].call_args_list]
        self.assertEqual(cleared, [f"{self.tenant()}-shop", f"{self.tenant()}-shop--b"])
        self.assertEqual(forget.call_args.args[2], f"{self.tenant()}-shop")
        self.fake["delete_hook"].assert_called_once_with("acme/shop", "tok", 77)
        self.assertNotIn(rkey, self.store)


class RemindersTest(AppTest):
    endpoint = "https://cliq.zoho.com/api/v2/applications/8855/incoming?appkey=abc123"

    def test_only_cliq_webhook_urls(self):
        for bad in ("https://evil.example/api/v2/applications/1/incoming?appkey=x",
                    "http://cliq.zoho.com/api/v2/applications/1/incoming?appkey=x", ""):
            resp = self.client.post("/reminders/connect", headers=self.headers(),
                                    json={"endpoint": bad, "token": "t"})
            self.assertEqual(resp.status_code, 400, bad)

    def test_connect_saves_after_a_successful_ping(self):
        with mock.patch.object(self.app, "ping_org", return_value=200), \
                mock.patch.object(self.app.registry, "save") as save:
            resp = self.client.post("/reminders/connect", headers=self.headers(),
                                    json={"endpoint": self.endpoint, "token": "t"})
        self.assertEqual(resp.get_json(), {"connected": True, "host": "cliq.zoho.com"})
        self.assertEqual(save.call_args.args[3], self.tenant())

    def test_tries_other_data_centres(self):
        codes = iter([401, 200])
        with mock.patch.object(self.app, "ping_org", side_effect=lambda *a: next(codes)), \
                mock.patch.object(self.app.registry, "save") as save:
            resp = self.client.post("/reminders/connect", headers=self.headers(),
                                    json={"endpoint": self.endpoint, "token": "t"})
        self.assertEqual(resp.get_json()["host"], "cliq.zoho.in")
        self.assertIn("cliq.zoho.in", save.call_args.args[4])

    def test_rejected_everywhere(self):
        with mock.patch.object(self.app, "ping_org", return_value=401), \
                mock.patch.object(self.app.registry, "save") as save:
            resp = self.client.post("/reminders/connect", headers=self.headers(),
                                    json={"endpoint": self.endpoint, "token": "t"})
        self.assertEqual(resp.status_code, 422)
        save.assert_not_called()

    def test_tick_pings_every_org(self):
        orgs = [{"tenant": "o1", "endpoint": "https://a", "token": "x"},
                {"tenant": "o2", "endpoint": "https://b", "token": "y"}]
        with mock.patch.object(self.app.registry, "load", return_value=orgs), \
                mock.patch.object(self.app, "ping_org", side_effect=[200, 500]):
            body = self.client.get("/tick", headers={"X-Conduit-Secret": support.SECRET}).get_json()
        self.assertEqual(body, {"orgs": 2, "delivered": 1})


if __name__ == "__main__":
    unittest.main()
