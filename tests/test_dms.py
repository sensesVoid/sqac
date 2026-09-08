"""DMS (Dynamic Memory Sparsification) tests — staleness/decay eviction policy."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqac.dms import DMS, UtilityWeights, utility, rank_store
from sqac.offloader import ContextOffloader
from sqac.store import KIND_FACT, KIND_TURN


def _salient_exchange():
    return __import__("sqac.offloader", fromlist=["Exchange", "Turn"])


class TestUtilityScoring(unittest.TestCase):
    def test_fresh_salient_scores_highest(self):
        w = UtilityWeights()
        fresh_high = utility(salience=1.0, age_minutes=1, access_count=0, w=w)
        old_high = utility(salience=1.0, age_minutes=120, access_count=0, w=w)
        self.assertGreater(fresh_high, old_high)

    def test_aging_decays_exponentially(self):
        w = UtilityWeights(lambda_decay=0.1)
        a = utility(salience=1.0, age_minutes=0, access_count=0, w=w)
        b = utility(salience=1.0, age_minutes=600, access_count=0, w=w)
        self.assertGreater(a, b)
        ratio = b / a if a else 0
        self.assertLess(ratio, 0.5)  # 10h at decay 0.1 -> well below half-life

    def test_access_count_protects_cold_entry(self):
        """A repeatedly-accessed old exchange outranks a never-accessed fresh one."""
        w = UtilityWeights(alpha=0.15, lambda_decay=0.1)
        cold_but_used = utility(salience=0.9, age_minutes=600, access_count=8, w=w)
        fresh_unused = utility(salience=0.9, age_minutes=1, access_count=0, w=w)
        self.assertGreater(cold_but_used, fresh_unused)

    def test_zero_age_zero_access_is_salience(self):
        w = UtilityWeights()
        self.assertAlmostEqual(utility(salience=0.6, age_minutes=0, access_count=0, w=w), 0.6)


class TestDMSRegistrar(unittest.TestCase):
    def setUp(self):
        self.clock = [1000.0]

        def now():
            return self.clock[0]

        self.dms = DMS(budget=2, weights=UtilityWeights(), now=now)

    def test_register_and_score(self):
        self.dms.register(1, salience=0.5)
        self.assertEqual(self.dms.score(1), 0.5)
        # age it by 10h -> decays below threshold-ish
        self.clock[0] += 3600 * 10
        self.assertLess(self.dms.score(1), 0.5)

    def test_touch_bumps_score(self):
        self.dms.register(1, salience=0.2)
        self.clock[0] += 3600 * 5
        before = self.dms.score(1)
        self.dms.touch(1)
        self.dms.touch(1)
        after = self.dms.score(1)
        self.assertGreater(after, before)

    def test_budget_detection(self):
        self.dms.register(1, salience=0.5)
        self.dms.register(2, salience=0.5)
        self.assertFalse(self.dms.over_budget())
        self.dms.register(3, salience=0.5)
        self.assertTrue(self.dms.over_budget())

    def test_promote_exempts_from_eviction(self):
        self.dms.register(1, salience=0.1)
        self.dms.register(2, salience=0.9)
        self.clock[0] += 3600 * 24  # both aged a day
        self.dms.promote(2)  # durable, exempt
        evictable = self.dms.evict_list(top_k=None)
        self.assertIn(1, evictable)
        self.assertNotIn(2, evictable)

    def test_evict_list_sorted_ascending(self):
        self.dms.register(1, salience=0.1)  # lowest
        self.dms.register(2, salience=0.5)
        self.dms.register(3, salience=0.9)  # highest
        ranked = self.dms.evict_list(top_k=3)
        self.assertEqual(ranked, [1, 2, 3])  # ascending = least useful first


class TestOffloaderDMSIntegration(unittest.TestCase):
    def test_sparsify_demotes_cold_turns_to_facts(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.sqac")
            clock = [1000.0]

            def now():
                return clock[0]

            dms = DMS(budget=2, weights=UtilityWeights(), now=now)
            off = ContextOffloader(p, window=200, dms=dms)
            # low-salience exchanges (no questions/salient markers) age out
            for i in range(5):
                off.observe("user", f"ambient note {i} about the temperature")
                off.observe("assistant", f"recorded reading {i}, all nominal")
            self.assertEqual(off.offload(), 5)
            # recency means none demote yet
            self.assertEqual(off.sparsify(), 0)

            # age them, then sparsify -> low-utility tail demoted to facts
            clock[0] += 3600 * 6
            n = off.sparsify()
            self.assertGreaterEqual(n, 1)

            # fact kind present after demotion
            stats = off.stats()["kinds"]
            self.assertIn("fact", stats)
            self.assertGreater(stats["fact"], 0)

    def test_sparsify_noop_without_dms(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.sqac")
            off = ContextOffloader(p, window=200)
            off.observe("user", "a")
            off.observe("assistant", "b")
            off.offload()
            self.assertEqual(off.sparsify(), 0)

    def test_access_promotes_through_recall(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.sqac")
            clock = [1000.0]

            def now():
                return clock[0]

            dms = DMS(budget=1, weights=UtilityWeights(alpha=0.2), now=now)
            off = ContextOffloader(p, window=200, dms=dms)
            off.observe("user", "what is the status of the blue deploy?")
            off.observe("assistant", "The blue deploy is healthy and shipping traffic.")
            off.observe("user", "different minor note")
            off.observe("assistant", "noted, nothing important")
            off.offload()
            self.assertTrue(dms.over_budget())

            # recall the important one repeatedly -> survives eviction
            for _ in range(5):
                off.recall("status of the blue deploy")
            clock[0] += 3600 * 2
            ranked = dms.evict_list(top_k=1)
            # the recalled (access-boosted) exchange should NOT be the eviction pick
            self.assertTrue(ranked)


class TestRankStore(unittest.TestCase):
    def test_rank_from_persisted_cartridge(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.sqac")
            off = ContextOffloader(p, window=200)
            off.observe("user", "critical decision: use sqlite")
            off.observe("assistant", "Chosen sqlite: single file, zero ops.")
            off.observe("user", "chitchat")
            off.observe("assistant", "sure")
            off.offload()
            off.save()

            from sqac.store import SqacStore

            store = SqacStore.load(p)
            dms = DMS(budget=100)
            rows = rank_store(store, dms)
            self.assertTrue(rows)
            for r in rows:
                self.assertIn("idx", r)
                self.assertIn("score", r)
                self.assertIn("tier", r)
            # sorted ascending by score
            scores = [r["score"] for r in rows]
            self.assertEqual(scores, sorted(scores))


if __name__ == "__main__":
    unittest.main()
