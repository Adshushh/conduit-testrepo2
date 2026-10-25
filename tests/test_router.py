"""Routing: AI reply parsing, blame, people matching and owner scoring."""
import unittest
from unittest import mock

import support
from support import Response

import router


def hit(path, lines="1-40", text="code"):
    return {"path": path, "lines": lines, "text": text, "score": 0.5}


def blame_range(start, end, login="", email="", name="Dev", date=None):
    user = {"login": login} if login else None
    commit = {"author": {"name": name, "email": email, "user": user}}
    if date:
        commit["authoredDate"] = date
    return {"startingLine": start, "endingLine": end, "commit": commit}


def blame_payload(ranges, field="defaultBranchRef"):
    return {"data": {"repository": {field: {"target": {"blame": {"ranges": ranges}}}}}}


class ParseJsonTest(support.TestCase):
    def test_plain_json(self):
        self.assertEqual(router.parse_json('{"ranked": [1]}'), {"ranked": [1]})

    def test_code_fence(self):
        self.assertEqual(router.parse_json('```json\n{"note": "x"}\n```'), {"note": "x"})

    def test_text_around_json(self):
        self.assertEqual(router.parse_json('Sure! {"ranked": [0]} Hope that helps.'), {"ranked": [0]})

    def test_not_json(self):
        with self.assertRaises(ValueError):
            router.parse_json("no json here")


class ToIndexTest(support.TestCase):
    hits = [hit("src/a.py"), hit("src/ui/button.tsx"), hit("b.py")]

    def test_int(self):
        self.assertEqual(router.to_index(1, self.hits), 1)

    def test_out_of_range(self):
        self.assertIsNone(router.to_index(3, self.hits))
        self.assertIsNone(router.to_index(-1, self.hits))

    def test_bool_is_not_an_index(self):
        self.assertIsNone(router.to_index(True, self.hits))

    def test_bracketed_and_string_numbers(self):
        self.assertEqual(router.to_index("[2]", self.hits), 2)
        self.assertEqual(router.to_index("0", self.hits), 0)

    def test_file_name_or_path(self):
        self.assertEqual(router.to_index("button.tsx", self.hits), 1)
        self.assertEqual(router.to_index("src/a.py", self.hits), 0)
        self.assertIsNone(router.to_index("missing.py", self.hits))


class CleanNoteTest(support.TestCase):
    hits = [hit("src/auth/login.py"), hit("src/ui/favorite-button.component.ts")]

    def test_bracket_reference_becomes_file_name(self):
        note = router.clean_note("The bug is in [1].", self.hits)
        self.assertEqual(note, "The bug is in favorite-button.component.ts.")

    def test_parenthesised_reference(self):
        note = router.clean_note("Login fails ([0]) on submit.", self.hits)
        self.assertEqual(note, "Login fails (login.py) on submit.")

    def test_snippet_words_removed(self):
        note = router.clean_note("See snippet 0 and the token check (snippets 0, 1).", self.hits)
        self.assertNotIn("snippet", note.lower())

    def test_unknown_reference_dropped(self):
        self.assertEqual(router.clean_note("Broken in [7].", self.hits), "Broken in.")


class PickFilesTest(support.TestCase):
    def test_distinct_files_in_ranked_order(self):
        hits = [hit("a.py", "1-40"), hit("a.py", "33-72"), hit("b.py"), hit("c.py"), hit("d.py")]
        files = router.pick_files(hits, [3, 0])
        self.assertEqual([f["path"] for f in files], ["c.py", "a.py", "b.py"])

    def test_search_order_without_ai(self):
        hits = [hit("a.py"), hit("b.py")]
        self.assertEqual([f["path"] for f in router.pick_files(hits, [])], ["a.py", "b.py"])


class MatchCliqTest(support.TestCase):
    people = {"adshushh": "adesh@acme.com", "dev@acme.com": "dev.cliq@acme.com", "alice": "alice@acme.com"}

    def match(self, login="", email=""):
        return router.match_cliq({"login": login, "email": email}, self.people, set(self.people.values()))

    def test_by_login(self):
        self.assertEqual(self.match(login="adshushh"), "adesh@acme.com")

    def test_by_noreply_email(self):
        self.assertEqual(self.match(email="12345+alice@users.noreply.github.com"), "alice@acme.com")
        self.assertEqual(self.match(email="alice@users.noreply.github.com"), "alice@acme.com")

    def test_by_commit_email(self):
        self.assertEqual(self.match(email="dev@acme.com"), "dev.cliq@acme.com")

    def test_commit_email_is_a_linked_cliq_email(self):
        self.assertEqual(self.match(email="alice@acme.com"), "alice@acme.com")

    def test_nobody(self):
        self.assertEqual(self.match(login="stranger", email="s@else.com"), "")


class IsBotTest(support.TestCase):
    def test_bots(self):
        self.assertTrue(router.is_bot("dependabot[bot]", "", ""))
        self.assertTrue(router.is_bot("", "", "github-actions"))

    def test_person(self):
        self.assertFalse(router.is_bot("Adesh", "adesh@acme.com", "adshushh"))


class BlameFileTest(support.TestCase):
    def blame(self, payload, status=200, ref=None, start=10, end=30):
        with mock.patch.object(router.requests, "post", return_value=Response(status, payload)) as post:
            result = router.blame_file("acme", "app", "tok", "src/a.py", start, end, ref)
        return result, post

    def test_counts_only_lines_inside_the_chunk(self):
        ranges = [blame_range(1, 15, login="Alice"),       # 10-15 -> 6 lines
                  blame_range(16, 40, email="Bob@Acme.com"),  # 16-30 -> 15 lines
                  blame_range(50, 60, login="carol")]      # outside
        owners, _ = self.blame(blame_payload(ranges))
        self.assertEqual(owners["alice"]["lines"], 6)
        self.assertEqual(owners["bob@acme.com"]["lines"], 15)
        self.assertNotIn("carol", owners)

    def test_bots_are_skipped(self):
        owners, _ = self.blame(blame_payload([blame_range(10, 30, login="dependabot[bot]")]))
        self.assertEqual(owners, {})

    def test_same_author_adds_up(self):
        ranges = [blame_range(10, 12, login="alice"), blame_range(20, 21, login="alice")]
        owners, _ = self.blame(blame_payload(ranges))
        self.assertEqual(owners["alice"]["lines"], 5)

    def test_branch_query(self):
        payload = blame_payload([blame_range(10, 30, login="alice")], field="ref")
        owners, post = self.blame(payload, ref="develop")
        self.assertEqual(owners["alice"]["lines"], 21)
        self.assertEqual(post.call_args.kwargs["json"]["variables"]["ref"], "refs/heads/develop")

    def test_missing_branch(self):
        with self.assertRaises(router.BranchMissing):
            self.blame({"data": {"repository": {"ref": None}}}, ref="gone")

    def test_rejected_token(self):
        with self.assertRaises(router.RouteError) as ctx:
            self.blame({}, status=401)
        self.assertIn("401", str(ctx.exception))

    def test_no_blame_returned(self):
        with self.assertRaises(router.RouteError) as ctx:
            self.blame({"data": {"repository": None}, "errors": [{"message": "Could not resolve"}]})
        self.assertIn("Could not resolve", str(ctx.exception))


NOW = router.datetime(2026, 10, 1, tzinfo=router.timezone.utc)


class AgeFactorTest(support.TestCase):
    def test_new_code_counts_fully(self):
        self.assertEqual(router.age_factor("2026-10-01T00:00:00Z", NOW), 1.0)

    def test_half_after_the_half_life(self):
        self.assertAlmostEqual(router.age_factor("2026-07-03T00:00:00Z", NOW), 0.5)

    def test_quarter_after_two_half_lives(self):
        self.assertAlmostEqual(router.age_factor("2026-04-04T00:00:00Z", NOW), 0.25)

    def test_very_old_code_keeps_the_floor(self):
        self.assertEqual(router.age_factor("2020-01-01T00:00:00Z", NOW), router.RECENCY_FLOOR)

    def test_missing_or_bad_date_counts_fully(self):
        self.assertEqual(router.age_factor(None, NOW), 1.0)
        self.assertEqual(router.age_factor("", NOW), 1.0)
        self.assertEqual(router.age_factor("yesterday", NOW), 1.0)

    def test_future_date_counts_fully(self):
        self.assertEqual(router.age_factor("2027-01-01T00:00:00Z", NOW), 1.0)

    def test_date_without_zone(self):
        self.assertAlmostEqual(router.age_factor("2026-07-03T00:00:00", NOW), 0.5)


class BlameAgeTest(support.TestCase):
    def blame(self, ranges):
        with mock.patch.object(router.requests, "post", return_value=Response(200, blame_payload(ranges))) as post:
            owners = router.blame_file("acme", "app", "tok", "src/a.py", 1, 100, now=NOW)
        return owners, post

    def test_query_asks_for_the_commit_date(self):
        _, post = self.blame([blame_range(1, 10, login="alice")])
        self.assertIn("authoredDate", post.call_args.kwargs["json"]["query"])

    def test_recent_lines_outweigh_old_ones(self):
        # Adesh wrote 30 lines two years ago, Ravi rewrote 15 last month
        owners, _ = self.blame([blame_range(1, 30, login="adesh", date="2024-10-01T00:00:00Z"),
                                blame_range(31, 45, login="ravi", date="2026-09-01T00:00:00Z")])
        self.assertEqual(owners["adesh"]["lines"], 30)
        self.assertAlmostEqual(owners["adesh"]["weighted"], 6.0)
        self.assertEqual(owners["ravi"]["lines"], 15)
        self.assertGreater(owners["ravi"]["weighted"], 11)
        self.assertGreater(owners["ravi"]["weighted"], owners["adesh"]["weighted"])

    def test_one_author_with_old_and_new_lines(self):
        owners, _ = self.blame([blame_range(1, 10, login="alice", date="2026-10-01T00:00:00Z"),
                                blame_range(11, 20, login="alice", date="2020-01-01T00:00:00Z")])
        self.assertEqual(owners["alice"]["lines"], 20)
        self.assertAlmostEqual(owners["alice"]["weighted"], 12.0)


class RouteHelpers(support.TestCase):
    """route() with fake search and blame; no tests of its own."""

    hits = [hit("src/a.py", "1-40"), hit("src/b.py", "1-40"), hit("src/c.py", "1-40"), hit("src/d.py")]

    def run_route(self, blames, people=None, llm=None, ai=None, ref="", codeowners=""):
        req = {"repo": "acme/app", "namespace": "acme", "token": "tok", "title": "Login fails",
               "description": "Error after submit", "severity": "high", "people": people or {}, "ref": ref}

        def fake_blame(owner, repo, token, path, start, end, ref=None, timeout=None):
            value = blames.get(path, {})
            if isinstance(value, Exception):
                raise value
            return value

        rules = router.parse_codeowners(codeowners) if codeowners else []
        patches = [mock.patch.object(router, "search_chunks", return_value=list(self.hits)),
                   mock.patch.object(router, "blame_file", side_effect=fake_blame),
                   mock.patch.object(router, "fetch_codeowners",
                                     return_value=(".github/CODEOWNERS" if rules else "", rules))]
        if ai is not None:
            patches.append(mock.patch.object(router, "llm_triage", return_value=ai))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return router.route(req, "fake-host", "key", llm)

    @staticmethod
    def owner(lines, login="", email=""):
        return {"lines": lines, "name": login or email, "login": login, "email": email}


class RouteTest(RouteHelpers):
    """route() end to end."""

    def test_score_is_lines_times_file_weight(self):
        blames = {"src/a.py": {"alice": self.owner(10, login="alice")},   # 10 x 3 = 30
                  "src/b.py": {"bob": self.owner(20, login="bob")},       # 20 x 2 = 40
                  "src/c.py": {"alice": self.owner(5, login="alice")}}    # 5 x 1 = 5
        result = self.run_route(blames)
        scores = {c["key"]: c["score"] for c in result["candidates"]}
        self.assertEqual(scores, {"alice": 35, "bob": 40})
        self.assertEqual(result["candidates"][0]["key"], "bob")
        self.assertEqual([f["path"] for f in result["files"]], ["src/a.py", "src/b.py", "src/c.py"])

    def test_people_mapping_merges_login_and_email(self):
        blames = {"src/a.py": {"alice": self.owner(10, login="alice")},
                  "src/b.py": {"alice@old.com": self.owner(10, email="alice@old.com")}}
        people = {"Alice": "Alice@Acme.com", "alice@old.com": "alice@acme.com"}
        result = self.run_route(blames, people=people)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["cliq"], "alice@acme.com")
        self.assertEqual(result["candidates"][0]["score"], 10 * 3 + 10 * 2)

    def test_unlinked_author_has_no_cliq_user(self):
        result = self.run_route({"src/a.py": {"zed": self.owner(3, login="zed")}})
        self.assertEqual(result["candidates"][0]["cliq"], "")

    def test_ai_ranking_changes_file_order(self):
        ai = {"ranked": [2, 0], "note": "Token check in c.py.", "questions": ["Which browser?"], "model": "m"}
        result = self.run_route({}, ai=ai)
        self.assertEqual([f["path"] for f in result["files"]], ["src/c.py", "src/a.py", "src/b.py"])
        self.assertTrue(result["ai_used"])
        self.assertEqual(result["note"], "Token check in c.py.")
        self.assertEqual(result["questions"], ["Which browser?"])

    def test_ai_error_falls_back_to_search_order(self):
        result = self.run_route({}, ai={"error": "AI call failed (503)"})
        self.assertFalse(result["ai_used"])
        self.assertEqual(result["files"][0]["path"], "src/a.py")
        self.assertTrue(any("AI call failed" in w for w in result["warnings"]))

    def test_blame_error_becomes_a_warning(self):
        blames = {"src/a.py": router.RouteError("GitHub blame failed (502)"),
                  "src/b.py": {"bob": self.owner(4, login="bob")}}
        result = self.run_route(blames)
        self.assertIn("GitHub blame failed (502)", result["warnings"])
        self.assertEqual(result["candidates"][0]["key"], "bob")

    def test_no_search_hits(self):
        req = {"repo": "acme/app", "namespace": "acme", "title": "x", "description": "y"}
        with mock.patch.object(router, "search_chunks", return_value=[]):
            with self.assertRaises(router.RouteError):
                router.route(req, "fake-host", "key")

    def test_bad_branch_name_is_ignored(self):
        result = self.run_route({}, ref="bad branch; rm -rf")
        self.assertEqual(result["debug"]["ref"], "(default branch)")

    def test_file_links(self):
        result = self.run_route({}, ref="develop")
        self.assertEqual(result["files"][0]["url"], "https://github.com/acme/app/blob/develop/src/a.py#L1-L40")

    def test_token_is_never_in_the_result(self):
        result = self.run_route({})
        self.assertNotIn("tok", str(result["debug"]["token"]))


class RecencyRouteTest(RouteHelpers):
    """route() scores by weighted lines but judges thin evidence by real lines."""

    @staticmethod
    def aged(lines, weighted, login):
        return {"lines": lines, "weighted": weighted, "name": login, "login": login, "email": ""}

    def test_recent_author_wins(self):
        blames = {"src/a.py": {"adesh": self.aged(30, 6.0, "adesh"), "ravi": self.aged(15, 12.8, "ravi")}}
        result = self.run_route(blames)
        self.assertEqual([c["key"] for c in result["candidates"]], ["ravi", "adesh"])
        scores = {c["key"]: c["score"] for c in result["candidates"]}
        self.assertEqual(scores, {"ravi": 38.4, "adesh": 18})

    def test_share_follows_the_weighted_score(self):
        blames = {"src/a.py": {"adesh": self.aged(30, 2.5, "adesh"), "ravi": self.aged(10, 7.5, "ravi")}}
        result = self.run_route(blames)
        shares = {c["key"]: c["share"] for c in result["candidates"]}
        self.assertEqual(shares, {"ravi": 75, "adesh": 25})

    def test_old_code_is_not_thin_evidence(self):
        # 40 real lines, all old: shares are not capped even though the weighted total is small
        blames = {"src/a.py": {"adesh": self.aged(40, 8.0, "adesh")}}
        result = self.run_route(blames)
        self.assertEqual(result["candidates"][0]["share"], 100)


class ConfidenceTest(RouteHelpers):
    """Two-part confidence: where (AI about the files) and who (share of the blamed code)."""

    def ai(self, level, reason="The token check in a.py matches the error."):
        return {"ranked": [0], "note": "", "questions": [], "model": "m",
                "code_level": level, "code_reason": reason}

    def by_key(self, result):
        return {c["key"]: c for c in result["candidates"]}

    def test_one_author_and_sure_ai(self):
        result = self.run_route({"src/a.py": {"alice": self.owner(30, login="alice")}}, ai=self.ai("high"))
        alice = self.by_key(result)["alice"]
        self.assertEqual(alice["share"], 100)
        self.assertEqual(alice["confidence"], 90)          # capped by "where": high = 90
        self.assertEqual(result["code_confidence"]["source"], "ai")
        self.assertEqual(result["code_confidence"]["reason"], "The token check in a.py matches the error.")

    def test_shared_code_is_a_close_call(self):
        blames = {"src/a.py": {"alice": self.owner(13, login="alice"), "bob": self.owner(12, login="bob")}}
        result = self.run_route(blames, ai=self.ai("high"))
        cands = self.by_key(result)
        self.assertEqual((cands["alice"]["share"], cands["bob"]["share"]), (52, 48))
        self.assertEqual(cands["alice"]["confidence"], 52)  # "who" is the weak part

    def test_vague_report_caps_confidence(self):
        result = self.run_route({"src/a.py": {"alice": self.owner(30, login="alice")}}, ai=self.ai("low"))
        self.assertEqual(self.by_key(result)["alice"]["confidence"], 35)  # "where" is the weak part
        self.assertEqual(result["code_confidence"]["level"], "low")

    def test_unlinked_author_lowers_the_linked_share(self):
        blames = {"src/a.py": {"zed": self.owner(21, login="zed"), "alice": self.owner(9, login="alice")}}
        result = self.run_route(blames, people={"alice": "alice@acme.com"}, ai=self.ai("high"))
        alice = self.by_key(result)["alice"]
        self.assertEqual(alice["cliq"], "alice@acme.com")
        self.assertEqual(alice["share"], 30)
        self.assertEqual(alice["confidence"], 30)

    def test_share_uses_weighted_scores_across_files(self):
        blames = {"src/a.py": {"alice": self.owner(10, login="alice")},   # 30
                  "src/b.py": {"bob": self.owner(10, login="bob")}}       # 20
        cands = self.by_key(self.run_route(blames, ai=self.ai("high")))
        self.assertEqual((cands["alice"]["share"], cands["bob"]["share"]), (60, 40))

    def test_thin_evidence_is_capped(self):
        result = self.run_route({"src/a.py": {"alice": self.owner(3, login="alice")}}, ai=self.ai("high"))
        self.assertTrue(result["thin_evidence"])
        self.assertEqual(result["blamed_lines"], 3)
        self.assertEqual(self.by_key(result)["alice"]["confidence"], router.THIN_CAP)

    def test_without_ai_uses_the_search_value(self):
        result = self.run_route({"src/a.py": {"alice": self.owner(30, login="alice")}},
                                ai={"error": "AI call failed (503)"})
        self.assertEqual(result["code_confidence"]["source"], "search")
        self.assertEqual(self.by_key(result)["alice"]["confidence"], router.NO_AI_CONFIDENCE)

    def test_ai_without_a_level(self):
        result = self.run_route({"src/a.py": {"alice": self.owner(30, login="alice")}}, ai=self.ai(""))
        self.assertEqual(result["code_confidence"]["source"], "search")
        self.assertIn("no confidence", result["code_confidence"]["reason"])

    def test_nobody_blamed(self):
        result = self.run_route({}, ai=self.ai("high"))
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["blamed_lines"], 0)


class CodeownersTest(RouteHelpers):
    """CODEOWNERS: parsing, GitHub-style matching and the owner bonus."""

    def test_parse_skips_teams_and_comments(self):
        rules = router.parse_codeowners("# owners\n*.js @Alice @acme/web  # team skipped\n/docs/ dev@acme.com\n")
        self.assertEqual(rules, [("*.js", ["alice"]), ("/docs/", ["dev@acme.com"])])

    def test_matching(self):
        match = router.codeowners_match
        self.assertTrue(match("*", "src/a.py"))
        self.assertTrue(match("*.js", "static/menu.js"))
        self.assertFalse(match("*.js", "app.py"))
        self.assertTrue(match("static/", "static/js/menu.js"))      # a folder at any depth
        self.assertTrue(match("/static/", "static/menu.js"))
        self.assertFalse(match("/static/", "web/static/menu.js"))   # anchored to the root
        self.assertTrue(match("src/*.py", "src/a.py"))
        self.assertFalse(match("src/*.py", "src/deep/a.py"))
        self.assertTrue(match("src/**/a.py", "src/deep/er/a.py"))
        self.assertTrue(match("/app.py", "app.py"))
        self.assertFalse(match("/app.py", "app.pyc"))               # whole names only
        self.assertFalse(match("src/a", "src/ab.py"))
        self.assertTrue(match("templates", "templates/menu.html"))

    def test_last_matching_rule_wins(self):
        rules = router.parse_codeowners("* @alice\nsrc/b.py @bob\n")
        self.assertEqual(router.owners_for(rules, "src/b.py"), ["bob"])
        self.assertEqual(router.owners_for(rules, "src/a.py"), ["alice"])

    def test_owner_bonus_breaks_a_close_call(self):
        blames = {"src/a.py": {"alice": self.owner(13, login="alice"), "bob": self.owner(12, login="bob")}}
        result = self.run_route(blames, codeowners="src/a.py @bob\n")
        scores = {c["key"]: c["score"] for c in result["candidates"]}
        self.assertEqual(scores, {"alice": 39, "bob": 36 + router.CODEOWNER_BONUS * 3})
        self.assertEqual(result["candidates"][0]["key"], "bob")
        self.assertTrue(result["candidates"][0]["codeowner"])
        self.assertFalse(result["candidates"][1]["codeowner"])
        self.assertEqual(result["debug"]["codeowners"]["files"], {"src/a.py": ["bob"]})

    def test_owner_without_blamed_lines_is_still_a_candidate(self):
        result = self.run_route({"src/a.py": {"alice": self.owner(30, login="alice")}},
                                people={"carol": "carol@acme.com"}, codeowners="*.py @carol\n")
        carol = [c for c in result["candidates"] if c["key"] == "carol"][0]
        self.assertEqual(carol["cliq"], "carol@acme.com")
        self.assertEqual(carol["score"], router.CODEOWNER_BONUS * (3 + 2 + 1))

    def test_bonus_does_not_count_as_blamed_lines(self):
        result = self.run_route({"src/a.py": {"alice": self.owner(3, login="alice")}}, codeowners="* @bob\n")
        self.assertEqual(result["blamed_lines"], 3)
        self.assertTrue(result["thin_evidence"])

    def test_no_codeowners_file(self):
        result = self.run_route({"src/a.py": {"alice": self.owner(10, login="alice")}})
        self.assertEqual(result["debug"]["codeowners"], {"source": "", "files": {}})
        self.assertFalse(result["candidates"][0]["codeowner"])


class FetchCodeownersTest(support.TestCase):
    def setUp(self):
        super().setUp()
        router._codeowners_cache.clear()

    def test_finds_the_first_location_and_caches_it(self):
        replies = [Response(404, {}), Response(200, {}, text="* @alice")]
        with mock.patch.object(router.requests, "get", side_effect=replies) as get:
            self.assertEqual(router.fetch_codeowners("acme/app", "tok"), ("CODEOWNERS", [("*", ["alice"])]))
            self.assertEqual(router.fetch_codeowners("acme/app", "tok"), ("CODEOWNERS", [("*", ["alice"])]))
        self.assertEqual(get.call_count, 2)                 # the second call came from the cache
        self.assertIn("/contents/.github/CODEOWNERS", get.call_args_list[0].args[0])

    def test_none_anywhere(self):
        with mock.patch.object(router.requests, "get", return_value=Response(404, {})) as get:
            self.assertEqual(router.fetch_codeowners("acme/app", "tok"), ("", []))
        self.assertEqual(get.call_count, 3)

    def test_network_error_is_not_cached(self):
        with mock.patch.object(router.requests, "get", side_effect=router.requests.ConnectionError()):
            self.assertEqual(router.fetch_codeowners("acme/app", "tok"), ("", []))
        self.assertNotIn(("acme/app", ""), router._codeowners_cache)

    def test_branch_is_passed(self):
        with mock.patch.object(router.requests, "get", return_value=Response(404, {})) as get:
            router.fetch_codeowners("acme/app", "tok", "develop")
        self.assertEqual(get.call_args.kwargs["params"], {"ref": "develop"})


class LlmTriageTest(support.TestCase):
    hits = [hit("src/a.py"), hit("src/b.py")]
    llm = {"key": "k", "base": "https://llm.example/v1", "model": "openai/gpt-oss-120b", "quickml": {}}

    def triage(self, response, ask=True):
        with mock.patch.object(router.requests, "post", return_value=response) as post:
            return router.llm_triage(self.llm, "t", "d", "high", self.hits, ask), post

    def test_off_without_a_key(self):
        self.assertIsNone(router.llm_triage({"key": "", "quickml": {}}, "t", "d", "", self.hits, True))

    def test_reads_the_reply(self):
        content = '{"ranked": ["[1]", 0, 0, 9], "note": "See [1].", "questions": ["Steps?", ""]}'
        reply = Response(200, {"choices": [{"message": {"content": content}}]})
        result, post = self.triage(reply)
        self.assertEqual(result["ranked"], [1, 0])
        self.assertEqual(result["note"], "See b.py.")
        self.assertEqual(result["questions"], ["Steps?"])
        self.assertEqual(post.call_args.kwargs["timeout"], router.LLM_TIMEOUT)
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "low")

    def test_reads_the_confidence(self):
        content = '{"ranked": [0], "note": "", "questions": [], "confidence": "Low", "confidence_reason": "Vague; see [1]."}'
        result, post = self.triage(Response(200, {"choices": [{"message": {"content": content}}]}))
        self.assertEqual(result["code_level"], "low")
        self.assertEqual(result["code_reason"], "Vague; see b.py.")
        self.assertIn('"confidence"', post.call_args.kwargs["json"]["messages"][1]["content"])

    def test_unknown_confidence_is_ignored(self):
        content = '{"ranked": [0], "note": "", "questions": [], "confidence": "very sure"}'
        result, _ = self.triage(Response(200, {"choices": [{"message": {"content": content}}]}))
        self.assertEqual(result["code_level"], "")

    def test_no_questions_when_not_asked(self):
        content = '{"ranked": [0], "note": "", "questions": ["Steps?"]}'
        result, _ = self.triage(Response(200, {"choices": [{"message": {"content": content}}]}), ask=False)
        self.assertEqual(result["questions"], [])

    def test_http_error(self):
        result, _ = self.triage(Response(503, {}))
        self.assertIn("503", result["error"])

    def test_reply_not_json(self):
        result, _ = self.triage(Response(200, {"choices": [{"message": {"content": "no idea"}}]}))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
