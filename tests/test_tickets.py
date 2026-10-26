"""Duplicate detection: which open tickets count as the same bug."""
import unittest
from unittest import mock

import support
from support import Response

import tickets


def search_reply(*hits):
    return {"result": {"hits": [{"_id": rid, "_score": score, "fields": {"ticket_id": tid, "title": title}}
                                for rid, score, tid, title in hits]}}


class SimilarTest(support.TestCase):
    def similar(self, response, **kwargs):
        with mock.patch.object(tickets.requests, "post", return_value=response) as post:
            return tickets.similar("host", "key", "o123-shop", "Login fails", "after submit", **kwargs), post

    def test_only_matches_above_the_threshold(self):
        reply = search_reply(("r1", 0.95, "SHOP-1", "Login broken"), ("r2", 0.80, "SHOP-2", "Slow page"))
        matches, _ = self.similar(Response(200, reply), threshold=0.87)
        self.assertEqual([m["ticket_id"] for m in matches], ["SHOP-1"])
        self.assertEqual(matches[0]["score"], 0.95)

    def test_excludes_the_ticket_itself(self):
        reply = search_reply(("r1", 0.99, "SHOP-1", "Same"), ("r2", 0.90, "SHOP-2", "Other"))
        matches, _ = self.similar(Response(200, reply), threshold=0.87, exclude="r1")
        self.assertEqual([m["record_id"] for m in matches], ["r2"])

    def test_reports_best_score_even_below_threshold(self):
        info = {}
        self.similar(Response(200, search_reply(("r1", 0.81234, "SHOP-1", "x"))), threshold=0.87, info=info)
        self.assertEqual(info["best"], 0.812)

    def test_at_most_three(self):
        reply = search_reply(*[(f"r{i}", 0.9, f"S-{i}", "x") for i in range(5)])
        matches, _ = self.similar(Response(200, reply))
        self.assertEqual(len(matches), 3)

    def test_no_tickets_stored_yet(self):
        matches, _ = self.similar(Response(404, {}))
        self.assertEqual(matches, [])

    def test_pinecone_error(self):
        with self.assertRaises(tickets.index_repo.IndexerError):
            self.similar(Response(500, {}, text="boom"))

    def test_searches_the_ticket_namespace(self):
        _, post = self.similar(Response(200, search_reply()))
        self.assertIn("/records/namespaces/tickets-o123-shop/search", post.call_args.args[0])


class NamespaceTest(support.TestCase):
    def test_prefix_and_length_limit(self):
        self.assertEqual(tickets.namespace_for("shop"), "tickets-shop")
        self.assertEqual(len(tickets.namespace_for("x" * 100)), 64)


class AddRemoveTest(support.TestCase):
    def test_add_stores_title_and_text(self):
        with mock.patch.object(tickets.index_repo, "upsert") as upsert:
            tickets.add("host", "key", "shop", 42, "SHOP-7", "Login fails", "after submit")
        host, key, namespace, records = upsert.call_args.args
        self.assertEqual(namespace, "tickets-shop")
        self.assertEqual(records[0]["_id"], "42")
        self.assertEqual(records[0]["ticket_id"], "SHOP-7")
        self.assertEqual(records[0][tickets.index_repo.TEXT_FIELD], "Login fails. after submit")

    def test_remove_treats_missing_as_done(self):
        for status, expected in ((200, True), (404, True), (500, False)):
            with mock.patch.object(tickets.requests, "post", return_value=Response(status, {})):
                self.assertEqual(tickets.remove("host", "key", "shop", "42"), expected)


if __name__ == "__main__":
    unittest.main()
