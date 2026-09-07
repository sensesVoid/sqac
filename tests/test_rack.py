"""CartridgeRack + graduation pass tests.

Pins the multi-cartridge contract:
  - multiple named .sqac cartridges mount into one rack
  - write_routed() sends a memory to the cartridge mapped for its kind
  - search() spans the rack and tags each hit with its cartridge
  - search_grouped() spans the rack, group-aware
  - graduation promotes high-salience session exchanges to kind=fact,
    honors an LLM-style grader callback, and is rerunnable (dedup)
  - the rack is the graduation target (rack.graduate -> default cartridge)
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.offloader import ContextOffloader
from sqac.rack import (
    CartridgeRack,
    GradedExchange,
    RackError,
    _gather_exchanges,
    default_grader,
    graduation_pass,
)
from sqac.store import KIND_FACT, KIND_TURN

# Salient session: bug diagnosis + a decision should graduate;
# chitchat and a bare "switching topics" should not.
TRANSCRIPT = [
    ("user", "Our CI is failing with a 504 on the deploy step."),
    ("assistant", "The 504 is the docker build timing out after 300 seconds."),
    ("user", "what is the root cause of that base image growth?"),
    ("assistant", "Base image grew to 2.1GB after adding the GPU toolchain; decision: slim it down next sprint."),
    ("user", "hey, how are you?"),
    ("assistant", "Doing great, ready to help!"),
    ("user", "switching topics, need the release notes drafted."),
    ("assistant", "Release notes done, covering the new export API."),
]


def _make_bucket(tmpdir: str, transcript=TRANSCRIPT) -> ContextOffloader:
    off = ContextOffloader(
        os.path.join(tmpdir, "session.sqac"), window=8, semantic=False
    )
    for role, text in transcript:
        off.observe(role, text)
    off.offload()
    return off


def _make_rack(tmpdir: str, routes=None, default="team") -> CartridgeRack:
    rack = CartridgeRack(tmpdir, routes=routes, default=default, semantic=False)
    return rack


class TestRackMount(unittest.TestCase):
    def test_create_register_contains(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            self.assertIn("team", rack)
            self.assertEqual(rack.names(), ["team"])
            # register from an existing file by path
            rack.save()
            rack2 = _make_rack(td)
            rack2.register("team")
            self.assertIn("team", rack2)

    def test_auto_load_mounts_existing_cartridges(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("alpha")
            rack["alpha"].add("Alpha holds deploy rules", key="rules")
            rack.create("beta")
            rack.save()

            rack2 = CartridgeRack(td, semantic=False)  # auto_load on by default
            self.assertIn("alpha", rack2)
            self.assertIn("beta", rack2)
            self.assertTrue(rack2.search("deploy rules"))

    def test_auto_load_skips_session_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            rack.save()
            off = ContextOffloader(
                os.path.join(td, "session.sqac"), window=8, semantic=False
            )
            off.observe("user", "turn memory, not durable")
            off.save()

            rack2 = CartridgeRack(td, semantic=False)
            self.assertNotIn("session", rack2)
            self.assertIn("team", rack2)

    def test_create_loads_existing_never_wipes(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            rack["team"].add("Do not wipe this", key="durable")
            rack.save()

            rack2 = _make_rack(td)
            rack2.create("team")  # file exists -> loaded, not reset
            self.assertEqual(len(rack2["team"]), 1)
            self.assertTrue(rack2.search("do not wipe"))

    def test_create_overwrite_resets(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            rack["team"].add("To be reset", key="x")
            rack.save()

            rack2 = _make_rack(td)
            rack2.create("team", overwrite=True)
            self.assertEqual(len(rack2["team"]), 0)

    def test_register_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            with self.assertRaises(RackError):
                rack.register("ghost")

    def test_unload_and_drop(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            rack.create("skills")
            self.assertTrue(rack.unload("skills"))
            self.assertNotIn("skills", rack)
            self.assertFalse(rack.unload("skills"))
            # drop deletes nothing by default
            self.assertTrue(rack.drop("team"))
            self.assertNotIn("team", rack)

    def test_require_missing_raises(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            with self.assertRaises(RackError):
                rack["nope"]

    def test_write_to_named_cartridge(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            rack.write("team", "Team Atlas owns the payments service", key="payments ownership")
            hits = rack.search("payments ownership")
            self.assertEqual(len(hits), 1)
            self.assertIn("Team Atlas", hits[0].content)


class TestRouting(unittest.TestCase):
    def test_write_routed_uses_route_map(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td, routes={"fact": "team", "skill": "skills"})
            rack.create("team")
            rack.create("skills")
            rack.write_routed("The deploy is ARM64 only", kind="fact")
            rack.write_routed(
                "SKILL weighted-index: label items 1..N, take i coins from item i",
                kind="skill",
            )
            # facts live only in team, skills only in skills
            team_hits = rack["team"].search("facts about deployment")
            skill_hits = rack["skills"].search("weighed index trick")
            self.assertTrue(team_hits or skill_hits)  # both non-empty by kinds below
            kinds = rack.stats()
            self.assertEqual(kinds["team"]["kinds"], {"fact": 1})
            self.assertEqual(kinds["skills"]["kinds"], {"skill": 1})

    def test_write_routed_unmapped_kind_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td, routes={"fact": "team"})
            rack.create("team")
            rack.write_routed("some doc knowledge", kind="doc")
            stats = rack.stats()["team"]
            self.assertEqual(stats["kinds"].get("doc"), 1)

    def test_write_routed_auto_creates_routed_cartridge(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td, routes={"fact": "missing"}, default=None)
            rack.write_routed("orphaned fact", kind="fact")
            self.assertIn("missing", rack)
            self.assertEqual(rack.stats()["missing"]["kinds"], {"fact": 1})

    def test_write_routed_auto_creates_default_cartridge(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td, routes={}, default="facts")
            rack.write_routed("some doc knowledge", kind="doc")
            self.assertIn("facts", rack)
            self.assertEqual(rack.stats()["facts"]["kinds"], {"doc": 1})

    def test_write_routed_no_route_or_default_raises(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td, routes={}, default=None)
            rack.create("team")
            with self.assertRaises(RackError):
                rack.write_routed("orphaned fact", kind="fact")


class TestRackSearch(unittest.TestCase):
    def test_search_merges_and_tags_cartridge(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td, routes={"fact": "team", "skill": "skills"})
            rack.create("team")
            rack.create("skills")
            rack["team"].add("Deploys are ARM64 only", key="deployment target")
            rack["skills"].add("SKILL: weighted-index trick", key="weighed index")
            hits = rack.search("what is the deployment target?")
            self.assertTrue(hits)
            self.assertEqual(hits[0].meta.get("cartridge"), "team")
            for h in rack.search("the weighed index trick"):
                self.assertIn(h.meta.get("cartridge"), ("team", "skills"))

    def test_search_kind_filter_spans_rack(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            rack.create("skills")
            rack["team"].add("Fact: deploy on ARM64", key="deploy arm64", kind="fact")
            rack["skills"].add(
                "SKILL: take i coins from item i", key="coins trick", kind="skill"
            )
            fact_hits = rack.search("deploy architecture", kind="fact")
            for h in fact_hits:
                self.assertEqual(h.meta.get("kind"), "fact")
            self.assertTrue(fact_hits)

    def test_search_grouped_across_rack(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("skills")
            g1 = rack["skills"]
            g1.add("trigger A1", key="weighted index a", meta={"skill": "weighted-index"})
            g1.add("trigger A2", key="weighed index alpha", meta={"skill": "weighted-index"})
            g1.add("unrelated", key="metric mass", meta={"skill": "metric-ma"})
            hits = rack.search_grouped("weighted index coins", group_key="skill")
            self.assertTrue(hits)
            self.assertEqual(hits[0].meta.get("skill"), "weighted-index")


class TestGraduationPass(unittest.TestCase):
    def test_gather_exchanges_dedupes_keys(self):
        with tempfile.TemporaryDirectory() as td:
            off = _make_bucket(td)
            exchanges = _gather_exchanges(off._store)
            # the 4 transcripts always produce exchanges: check dedupe by xid
            xids = [g.xid for g in exchanges]
            self.assertEqual(len(xids), len(set(xids)))
            self.assertGreaterEqual(len(xids), 4)

    def test_promotes_salient_skips_chitchat(self):
        with tempfile.TemporaryDirectory() as td:
            off = _make_bucket(td)
            target = CartridgeRack(td).create("team")["team"]
            report = graduation_pass(off._store, target, threshold=0.30)
            self.assertEqual(report["reviewed"], len(_gather_exchanges(off._store)))
            self.assertGreater(report["promoted"], 0)
            self.assertEqual(report["skipped"] + report["promoted"], report["reviewed"])
            # promoted entries are kind=fact with source lineage
            kinds = target.stats()["kinds"]
            self.assertEqual(kinds.get("fact", 0), report["promoted"])
            for e in target._entries:
                self.assertEqual(e["kind"], KIND_FACT)
                self.assertIn("graduated_from", e["meta"])
                self.assertNotIn("[turns", e["content"])  # durable text, not labels

    def test_default_grader_matches_threshold(self):
        g = default_grader(0.40)
        self.assertTrue(g(GradedExchange(xid=1, content="x", salience=0.8, turns=(0, 1))))
        self.assertFalse(g(GradedExchange(xid=2, content="x", salience=0.1, turns=(0, 1))))

    def test_custom_grader_override(self):
        # a grader that promotes EVERYTHING (count-based rules, LLM, ...)
        reviewed = []

        def keep_all(g: GradedExchange) -> bool:
            reviewed.append(g.xid)
            return True

        with tempfile.TemporaryDirectory() as td:
            off = _make_bucket(td)
            target = CartridgeRack(td).create("team")["team"]
            report = graduation_pass(off._store, target, grader=keep_all)
            self.assertEqual(report["promoted"], report["reviewed"])
            self.assertEqual(len(reviewed), report["reviewed"])

    def test_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            off = _make_bucket(td)
            rack = CartridgeRack(td)
            rack.create("team")
            r1 = rack.graduate(off._store, "team", threshold=0.30)
            r2 = rack.graduate(off._store, "team", threshold=0.30)
            self.assertEqual(r2["promoted"], 0, "second pass must not duplicate")
            self.assertEqual(r2["skipped"], r1["reviewed"])

    def test_rack_graduate_uses_default_target(self):
        with tempfile.TemporaryDirectory() as td:
            off = _make_bucket(td)
            rack = CartridgeRack(td, default="durable")
            rack.create("durable")
            rack.create("personal")
            report = rack.graduate(off._store, threshold=0.30)
            self.assertGreater(report["promoted"], 0)
            self.assertEqual(rack.stats()["durable"]["kinds"].get("fact", 0),
                             report["promoted"])
            self.assertNotIn("fact", rack.stats()["personal"]["kinds"])

    def test_rack_graduate_accepts_offloader_and_name(self):
        with tempfile.TemporaryDirectory() as td:
            off = _make_bucket(td)
            rack = CartridgeRack(td, default="durable")
            rack.create("durable")
            rack.create("bucket")
            # mount the session bucket as a named cartridge in the rack
            off.save()
            rack.register("bucket")
            # ContextOffloader object as source
            r1 = rack.graduate(off, target_name="durable", threshold=0.30)
            self.assertGreater(r1["promoted"], 0)
            # mounted cartridge NAME as source (same bucket: nothing new)
            r2 = rack.graduate("bucket", target_name="durable", threshold=0.30)
            self.assertEqual(r2["promoted"], 0)

    def test_graduated_facts_searchable_in_target(self):
        with tempfile.TemporaryDirectory() as td:
            off = _make_bucket(td)
            rack = CartridgeRack(td, default="durable")
            rack.create("durable")
            rack.graduate(off._store, threshold=0.30)
            # the 504 root cause question graduated; a back-reference finds it
            hits = rack.search("what was that 504 deploy timeout about?")
            self.assertTrue(hits)
            self.assertEqual(hits[0].meta.get("cartridge"), "durable")
            self.assertIn("504", hits[0].content)


class TestRackPersistence(unittest.TestCase):
    def test_save_then_register_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            rack["team"].add("Deploys are ARM64 only", key="deploy target")
            rack.save()
            fresh = _make_rack(td)
            fresh.register("team")
            hits = fresh.search("deploy architecture")
            self.assertTrue(hits)
            self.assertEqual(hits[0].meta.get("cartridge"), "team")

    def test_drop_with_path_deletes_file(self):
        with tempfile.TemporaryDirectory() as td:
            rack = _make_rack(td)
            rack.create("team")
            p = os.path.join(td, "team.sqac")
            self.assertTrue(os.path.exists(p))
            rack.drop("team", recursive_path=True)
            self.assertFalse(os.path.exists(p))


if __name__ == "__main__":
    unittest.main()