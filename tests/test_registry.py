"""Reminder registry: org tokens are stored encrypted and tampering is caught."""
import base64
import json
import unittest

import support

import registry


class SealTest(support.TestCase):
    data = {"endpoint": "https://cliq.zoho.com/api/v2/applications/1/incoming?appkey=x", "token": "t0k"}

    def test_round_trip(self):
        self.assertEqual(registry.unseal("secret", registry.seal("secret", self.data)), self.data)

    def test_token_is_not_readable(self):
        self.assertNotIn("t0k", base64.urlsafe_b64decode(registry.seal("secret", self.data)).decode("latin-1"))

    def test_each_seal_is_different(self):
        self.assertNotEqual(registry.seal("secret", self.data), registry.seal("secret", self.data))

    def test_wrong_secret(self):
        self.assertIsNone(registry.unseal("other", registry.seal("secret", self.data)))

    def test_tampered(self):
        # flip one bit inside the token: the JSON stays valid, so only the tag check can catch it
        raw = bytearray(base64.urlsafe_b64decode(registry.seal("secret", self.data)))
        raw[48 + json.dumps(self.data).index("t0k") + 1] ^= 1
        self.assertIsNone(registry.unseal("secret", base64.urlsafe_b64encode(bytes(raw)).decode()))

    def test_garbage(self):
        for blob in ("", "not base64!!", base64.urlsafe_b64encode(b"short").decode()):
            self.assertIsNone(registry.unseal("secret", blob))

    def test_record_id(self):
        self.assertEqual(registry.record_id(""), "host")
        self.assertEqual(registry.record_id("oabc123def"), "oabc123def")


if __name__ == "__main__":
    unittest.main()
