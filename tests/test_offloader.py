"""ContextOffloader tests — the context bucket contract.

Pins the measured design rules (prototype: verbatim turns 2/8 recall,
exchange indexing 6/8, denyxised keys 8/8):

  - observe() evicts complete exchanges beyond the window; buffer bounded
  - back-reference recall: "that 504 bug from earlier" -> right exchange
  - fail-safe: no confident hit -> "" / [] — never inject a guess
  - keys are denyxised at write time; thin keys get anchor substitution
  - bucket survives save/load; entries are kind=turn
  - custom distillers (LLM-backed) drop in cleanly
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.offloader import (
    ContextOffloader,
    Exchange,
    Turn,
    _DEIXIS,
    _salience,
    heuristic_distill,
)
from sqac.store import KIND_TURN

TRANSCRIPT = [
    ("user", "Our CI is failing with a 504 on the deploy step."),
    ("assistant", "The 504 comes from the deploy step timing out after 300 seconds while the docker build runs. The build got slower since the base image grew to 2.1GB."),
    ("user", "ok lets fix that later, what about the flaky unit tests?"),
    ("assistant", "The flaky tests are the websocket integration ones: they bind port 0 but then read the actual port from a shared file, which races under parallel runners."),
    ("user", "and the lint errors?"),
    ("assistant", "Lint failures are unused imports in three files from the refactor."),
    ("user", "switching topics: can you draft the release notes?"),
    ("assistant", "Release notes drafted, covering the new export API."),
    ("user", "what is our p99 latency on the search endpoint?"),
    ("assistant", "Search endpoint p99 is 180ms, measured from the last load test."),
    ("user", "summarize the meeting notes from yesterday"),
    ("assistant", "Meeting summary: agreed to migrate CI runners to ARM64."),
    ("user", "review this PR description"),
    ("assistant", "PR description reviewed, suggested tightening the migration section."),
]

# (query, substring that must appear in the top-1 recalled exchange)
BACKREF_CASES = [
    ("what was the cause of that 504 timeout we saw earlier?", "504"),
    ("that flaky websocket test problem from before, what was it?", "websocket"),
    ("why were the lint errors happening again?", "lint"),
    ("the port issue you mentioned, what was that about?", "websocket"),
    ("what did we say about the deploy step failing in CI?", "504"),
    ("earlier you mentioned something about the base image, what?", "base image"),
    ("what was the search endpoint latency you told me?", "180ms"),
    ("what did the meeting decide yesterday?", "arm64"),
]


def _seed(off: ContextOffloader) -> None:
    for role, text in TRANSCRIPT:
        off.observe(role, text)
    off.offload()


class TestOffloaderBasics(unittest.TestCase):
    def test_observe_evicts_and_bounds_buffer(self):
        off = ContextOffloader(window=8)
        evicted = []
        for role, text in TRANSCRIPT:
            r = off.observe(role, text)
            if r is not None:
                evicted.append(r)
            self.assertLessEqual(len(off._buffer), off.window)
        self.assertEqual(evicted, [1, 2, 3, 4])  # exchanges numbered from 1
        self.assertEqual(off.offload(), 3)       # flush the tail
        self.assertEqual(off.stats()["exchanges_offloaded"], 7)
        self.assertEqual(len(off._buffer), 0)

    def test_offload_flushes_everything(self):
        off = ContextOffloader(window=100)  # huge window: nothing auto-evicts
        for role, text in TRANSCRIPT:
            off.observe(role, text)
        self.assertEqual(len(off._buffer), len(TRANSCRIPT))
        self.assertEqual(off.offload(), 7)
        self.assertEqual(len(off._buffer), 0)

    def test_empty_turns_ignored(self):
        off = ContextOffloader()
        self.assertIsNone(off.observe("user", ""))
        self.assertIsNone(off.observe("user", "   "))
        self.assertEqual(len(off._buffer), 0)


class TestBackreferenceRecall(unittest.TestCase):
    """The flagship: deixis-laden queries must find the right exchange."""

    @classmethod
    def setUpClass(cls):
        if not cls._sem_available():
            raise unittest.SkipTest("semantic tier files unavailable")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "session.sqac")
        off = ContextOffloader(cls.path, window=8)
        _seed(off)
        off.save()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def _sem_available() -> bool:
        try:
            from sqac.static_encoder import StaticSimHashEncoder
            return StaticSimHashEncoder.available()
        except Exception:
            return False

    def test_backreference_recall_floor(self):
        off = ContextOffloader(self.path)
        ok = 0
        for q, mark in BACKREF_CASES:
            d = off.recall_detailed(q)
            good = bool(d) and mark in d[0]["content"].lower()
            ok += good
            if not good:
                self.fail(
                    f"recall miss: {q!r} -> "
                    f"{d[0]['content'][:60] if d else 'no confident hit'}"
                )
        self.assertEqual(ok, len(BACKREF_CASES))

    def test_recall_text_block_shape(self):
        off = ContextOffloader(self.path)
        text = off.recall("what was the cause of that 504 timeout we saw earlier?")
        self.assertTrue(text)
        self.assertIn("[turns", text)
        self.assertRegex(text, r"^\[0\.\d{2}\]")

    def test_recall_dedupes_exchange_keys(self):
        """One exchange has many keys; it must be injected at most once."""
        off = ContextOffloader(self.path)
        text = off.recall("what was the cause of that 504 timeout we saw earlier?")
        lines = [l for l in text.splitlines() if l.strip()]
        self.assertTrue(lines)
        self.assertEqual(len(set(lines)), len(lines), f"duplicate injection: {text!r}")

    def test_recall_failsafe_on_unrelated_query(self):
        off = ContextOffloader(self.path)
        self.assertEqual(off.recall("favorite ice cream flavor"), "")
        self.assertEqual(off.recall_detailed("favorite ice cream flavor"), [])

    def test_recall_detailed_fields(self):
        off = ContextOffloader(self.path)
        d = off.recall_detailed("the websocket flaky tests from before")
        self.assertTrue(d)
        for field in ("content", "confidence", "exchange", "turns", "salience", "mode"):
            self.assertIn(field, d[0])
        self.assertIn(d[0]["mode"], ("fuzzy", "semantic"))

    def test_entries_are_turn_kind(self):
        off = ContextOffloader(self.path)
        kinds = off.stats()["kinds"]
        self.assertEqual(kinds, {"turn": off.stats()["entries"]})


class TestDenyxis(unittest.TestCase):
    def test_keys_contain_no_deixis(self):
        ex = Exchange(
            xid=1,
            turns=[
                Turn("user", "can you fix that thing you mentioned earlier about the cache?"),
                Turn("assistant", "The cache layer needs invalidation on user update; keys stale for 60s."),
            ],
            turn_range=(0, 1),
        )
        keys, _ = heuristic_distill(ex)
        self.assertTrue(keys)
        for k in keys:
            words = set(k.lower().split())
            self.assertEqual(words & _DEIXIS, set(), f"deixis leaked into key: {k!r}")

    def test_thin_question_gets_anchor_substitution(self):
        ex = Exchange(
            xid=1,
            turns=[
                Turn("user", "and that?"),
                Turn("assistant", "The websocket tests bind port 0 and race on the shared port file."),
            ],
            turn_range=(0, 1),
        )
        keys, _ = heuristic_distill(ex)
        # the bare question denies to nearly nothing; anchors must rescue it
        joined = " ".join(keys)
        self.assertIn("websocket", joined)

    def test_distiller_injection(self):
        calls = []

        def llm_distill(ex: Exchange):
            calls.append(ex.xid)
            return ["LLM written key"], "LLM distilled content"

        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.sqac")
            off = ContextOffloader(p, window=4, distiller=llm_distill)
            off.observe("user", "question one")
            off.observe("assistant", "answer one")
            off.observe("user", "question two")
            off.observe("assistant", "answer two")
            off.offload()
            self.assertEqual(calls, [1, 2])
            off.save()
            back = ContextOffloader(p)
            text = back.recall("LLM written key")
            self.assertIn("LLM distilled content", text)


class TestPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TestBackreferenceRecall._sem_available():
            raise unittest.SkipTest("semantic tier files unavailable")

    def test_save_load_recall_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.sqac")
            off = ContextOffloader(p, window=8)
            _seed(off)
            n = off.stats()["exchanges_offloaded"]
            off.save()
            back = ContextOffloader(p)
            self.assertEqual(back.stats()["exchanges_offloaded"], n)
            d = back.recall_detailed("that 504 deploy timeout from earlier")
            self.assertTrue(d)
            self.assertIn("504", d[0]["content"])

    def test_save_without_path_raises(self):
        off = ContextOffloader(None)
        off.observe("user", "hi")
        with self.assertRaises(ValueError):
            off.save()


class TestFromTranscript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TestBackreferenceRecall._sem_available():
            raise unittest.SkipTest("semantic tier files unavailable")

    def test_transcript_ingest_with_junk_lines(self):
        with tempfile.TemporaryDirectory() as td:
            tr = os.path.join(td, "t.jsonl")
            lines = [
                json.dumps({"role": "user", "content": "we chose sqlite for the edge cache"}),
                json.dumps({"role": "assistant", "content": "Edge cache decision: sqlite, because single-file and zero-ops on devices."}),
                "not json at all",
                json.dumps({"role": "system"}),  # missing content: skipped
                json.dumps({"role": "user", "message": "why did we pick sqlite again?"}),
                json.dumps({"role": "assistant", "content": "Because it is single-file and zero-ops."}),
            ]
            tr_data = "\n".join(lines)
            with open(tr, "w", encoding="utf-8") as f:
                f.write(tr_data + "\n")
            p = os.path.join(td, "s.sqac")
            off = ContextOffloader.from_transcript(tr, p)
            self.assertGreater(off.stats()["exchanges_offloaded"], 0)
            d = off.recall_detailed("why did we pick sqlite for the edge?")
            self.assertTrue(d)
            self.assertIn("sqlite", d[0]["content"].lower())
            # message-field alias worked (second exchange exists)
            d2 = off.recall_detailed("single file zero ops reasoning")
            self.assertTrue(d2)


class TestSalience(unittest.TestCase):
    def test_salient_exchange_outscores_chitchat(self):
        chitchat = Exchange(
            xid=1,
            turns=[Turn("user", "hey there"), Turn("assistant", "Hello! How can I help?")],
            turn_range=(0, 1),
        )
        important = Exchange(
            xid=2,
            turns=[
                Turn("user", "what was the root cause?"),
                Turn("assistant", "The deploy fails with a 504 timeout after 300s; decision: slim the base image."),
            ],
            turn_range=(2, 3),
        )
        self.assertGreater(_salience(important), _salience(chitchat))


if __name__ == "__main__":
    unittest.main()
