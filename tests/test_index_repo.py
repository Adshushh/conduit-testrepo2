"""Indexing: which files are read and how they are cut into chunks."""
import unittest

import support

import index_repo


class ChunkFileTest(support.TestCase):
    def chunks(self, line_count):
        text = "\n".join(f"line {n}" for n in range(1, line_count + 1))
        return list(index_repo.chunk_file("src/app.py", text))

    def test_overlapping_line_ranges(self):
        chunks = self.chunks(100)
        self.assertEqual([c["lines"] for c in chunks], ["1-40", "33-72", "65-100"])

    def test_short_file_is_one_chunk(self):
        chunks = self.chunks(10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0]["start_line"], chunks[0]["end_line"]), (1, 10))

    def test_text_starts_with_path_and_is_capped(self):
        chunk = next(index_repo.chunk_file("a.py", "x" * 5000))
        self.assertTrue(chunk[index_repo.TEXT_FIELD].startswith("File: a.py\n"))
        self.assertLessEqual(len(chunk[index_repo.TEXT_FIELD]), index_repo.CHUNK_MAX_CHARS)

    def test_ids_are_stable_and_unique(self):
        first = [c["_id"] for c in self.chunks(100)]
        self.assertEqual(first, [c["_id"] for c in self.chunks(100)])
        self.assertEqual(len(set(first)), len(first))

    def test_blank_file(self):
        self.assertEqual(list(index_repo.chunk_file("a.py", "\n\n  \n")), [])


class ShouldSkipTest(support.TestCase):
    def test_code_files_are_kept(self):
        for path in ("src/app.py", "web/App.tsx", "handlers/bot.dg", "schema.prisma"):
            self.assertFalse(index_repo.should_skip(path.split("/")), path)

    def test_skipped(self):
        for path in ("node_modules/x/index.js", "dist/app.js", "package-lock.json",
                     "static/app.min.js", "logo.png", "src/__pycache__/a.py"):
            self.assertTrue(index_repo.should_skip(path.split("/")), path)


if __name__ == "__main__":
    unittest.main()
