"""Indexing big repos: the archive is streamed and chunks are uploaded in batches."""
import io
import tarfile
import unittest
from unittest import mock

import support

import index_repo


def tarball(files):
    """A GitHub-style .tar.gz: everything inside one top-level folder."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, data in files.items():
            raw = data if isinstance(data, bytes) else data.encode()
            info = tarfile.TarInfo(f"acme-shop-abc123/{path}")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


class StreamedResponse:
    """Like requests' streamed response: no .content, only pieces."""

    def __init__(self, body, status=200, piece=7):
        self.status_code = status
        self._body, self._piece = body, piece

    def iter_content(self, size):
        for i in range(0, len(self._body), self._piece):
            yield self._body[i:i + self._piece]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class DownloadTest(support.TestCase):
    def files(self, archive, status=200):
        with mock.patch.object(index_repo.requests, "get", return_value=StreamedResponse(archive, status)) as get:
            found = dict(index_repo.download_repo("acme/shop", "tok", "main"))
        return found, get

    def test_reads_files_from_a_streamed_archive(self):
        archive = tarball({"src/app.py": "print('hi')\n", "README.md": "# Shop\n",
                           "node_modules/x/index.js": "skip()", "logo.png": b"\x89PNG\x00",
                           "data/blob.py": b"\x00\x01binary", "big.py": "x" * (index_repo.MAX_FILE_BYTES + 1)})
        found, get = self.files(archive)
        self.assertEqual(sorted(found), ["README.md", "src/app.py"])
        self.assertEqual(found["src/app.py"], "print('hi')\n")
        self.assertTrue(get.call_args.kwargs["stream"])
        self.assertTrue(get.call_args.args[0].endswith("/tarball/main"))

    def test_github_errors(self):
        for status, words in ((401, "rejected the token"), (404, "can't read"), (500, "GitHub error 500")):
            with self.assertRaises(index_repo.IndexerError) as ctx:
                self.files(b"", status)
            self.assertIn(words, str(ctx.exception))


class RunIndexTest(support.TestCase):
    def run_index(self, files, every=3):
        uploads = []
        with mock.patch.object(index_repo.requests, "get", return_value=StreamedResponse(tarball(files))), \
                mock.patch.object(index_repo, "ensure_index", return_value="host") as ensure, \
                mock.patch.object(index_repo, "clear_namespace", return_value=True) as clear, \
                mock.patch.object(index_repo, "upsert", side_effect=lambda h, k, ns, recs: uploads.append(list(recs))), \
                mock.patch.object(index_repo, "UPLOAD_EVERY", every):
            progress = mock.Mock()
            result = index_repo.run_index("acme/shop", "ns", "tok", "key", progress=progress)
        return result, uploads, clear, ensure, progress

    def test_uploads_in_batches_while_reading(self):
        files = {f"src/f{i}.py": "x = 1\n" for i in range(7)}       # one chunk each
        (count, chunks, host), uploads, clear, ensure, progress = self.run_index(files, every=3)
        self.assertEqual((count, chunks, host), (7, 7, "host"))
        self.assertEqual([len(b) for b in uploads], [3, 3, 1])
        clear.assert_called_once_with("host", "key", "ns")
        self.assertEqual(progress.call_args_list[-1].args, ("done", 7, 7))

    def test_never_holds_more_than_a_batch(self):
        files = {f"src/f{i}.py": "line\n" * 100 for i in range(20)}
        per_file = len(list(index_repo.chunk_file("x.py", "line\n" * 100)))
        _, uploads, _, _, _ = self.run_index(files, every=10)
        self.assertTrue(all(len(b) < 10 + per_file for b in uploads))   # at most one file over a batch
        self.assertEqual(sum(len(b) for b in uploads), 20 * per_file)

    def test_nothing_to_index(self):
        with self.assertRaises(index_repo.IndexerError):
            self.run_index({"logo.png": b"\x89PNG\x00"})


if __name__ == "__main__":
    unittest.main()
