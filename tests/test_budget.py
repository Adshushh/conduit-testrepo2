"""The /route time budget: slow steps leave less time for the next, never past Cliq's wait."""
import unittest
from unittest import mock

import support
from support import Response

import quickml
import router


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class BudgetTest(support.TestCase):
    hits = [{"path": "src/a.py", "lines": "1-40", "text": "code", "score": 0.5}]
    groq = {"key": "k", "base": "https://llm.example/v1", "model": "m", "quickml": {}}
    both = {"key": "k", "base": "https://llm.example/v1", "model": "m",
            "quickml": {"url": "https://ml", "endpoint_key": "e", "org": "o", "client_id": "c",
                        "client_secret": "s", "refresh_token": "r", "accounts": "https://acc"}}
    reply = Response(200, {"choices": [{"message": {"content": '{"ranked": [0], "note": "x", "questions": []}'}}]})

    def setUp(self):
        super().setUp()
        self.clock = FakeClock()
        patcher = mock.patch.object(router, "clock", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def route(self, llm, search_seconds=0.0, blame=None):
        def slow_search(*args, **kwargs):
            self.clock.now += search_seconds
            return list(self.hits)

        blame = blame or mock.Mock(return_value={})
        req = {"repo": "acme/app", "namespace": "acme", "title": "t", "description": "d"}
        with mock.patch.object(router, "search_chunks", side_effect=slow_search), \
                mock.patch.object(router, "blame_file", blame), \
                mock.patch.object(router, "fetch_codeowners", return_value=("", [])):
            return router.route(req, "host", "key", llm), blame

    def test_fast_search_gets_the_ai(self):
        with mock.patch.object(router.requests, "post", return_value=self.reply) as post:
            result, _ = self.route(self.groq, search_seconds=1)
        self.assertTrue(result["ai_used"])
        self.assertEqual(post.call_args.kwargs["timeout"], router.LLM_TIMEOUT)

    def test_slow_search_skips_the_ai_but_still_routes(self):
        spent = router.ROUTE_BUDGET - router.BLAME_RESERVE - router.MIN_AI_SECONDS + 1
        with mock.patch.object(router.requests, "post") as post:
            result, blame = self.route(self.groq, search_seconds=spent)
        post.assert_not_called()
        self.assertFalse(result["ai_used"])
        self.assertTrue(any("AI skipped to answer in time" in w for w in result["warnings"]))
        blame.assert_called_once()

    def test_ai_timeout_shrinks_with_the_time_left(self):
        spent = router.ROUTE_BUDGET - router.BLAME_RESERVE - 5          # 5 seconds left for AI
        with mock.patch.object(router.requests, "post", return_value=self.reply) as post:
            self.route(self.groq, search_seconds=spent)
        self.assertAlmostEqual(post.call_args.kwargs["timeout"], 5)

    def test_no_fallback_when_quickml_used_the_time(self):
        def slow_quickml(cfg, prompt, timeout):
            self.clock.now += timeout
            raise quickml.QuickMLError("QuickML call failed (504)")

        with mock.patch.object(router.quickml, "generate", side_effect=slow_quickml), \
                mock.patch.object(router.requests, "post") as post:
            result, _ = self.route(self.both, search_seconds=1)
        post.assert_not_called()
        self.assertTrue(any("no time left for the fallback AI" in w for w in result["warnings"]))

    def test_fallback_runs_when_quickml_fails_fast(self):
        with mock.patch.object(router.quickml, "generate", side_effect=quickml.QuickMLError("down")), \
                mock.patch.object(router.requests, "post", return_value=self.reply) as post:
            result, _ = self.route(self.both, search_seconds=1)
        post.assert_called_once()
        self.assertTrue(result["ai_used"])

    def test_blame_gets_what_is_left_but_never_too_little(self):
        _, blame = self.route({}, search_seconds=router.ROUTE_BUDGET - 4)
        self.assertEqual(blame.call_args.args[7], 3)                       # 4 left minus 1, floor 3
        _, blame = self.route({}, search_seconds=router.ROUTE_BUDGET)
        self.assertEqual(blame.call_args.args[7], router.MIN_STEP_SECONDS)
        _, blame = self.route({}, search_seconds=0)
        self.assertEqual(blame.call_args.args[7], 12)

    def test_budget_in_debug(self):
        result, _ = self.route({}, search_seconds=2)
        self.assertEqual(result["debug"]["seconds"]["budget"], router.ROUTE_BUDGET)
        self.assertEqual(result["debug"]["seconds"]["search"], 2)


if __name__ == "__main__":
    unittest.main()
