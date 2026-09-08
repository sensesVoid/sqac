"""Tests for CartridgeRack auto-split and folder routing features.

Run: python -m pytest tests/test_rack_features.py -q
"""

import os
import tempfile
import unittest
from pathlib import Path

from sqac import CartridgeRack, SqacStore


class TestAutoSplit(unittest.TestCase):
    """Auto-split creates new shards when a cartridge hits max_entries."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_no_split_below_threshold(self):
        """Writes below the threshold stay in one cartridge."""
        rack = CartridgeRack(
            directory=self.tmp.name,
            default="memory",
            max_entries=10,
            auto_load=False,
        )
        rack.create("memory")
        for i in range(5):
            rack.write("memory", f"fact {i}", key=f"k{i}", kind="fact")
        rack.save()
        # Only one file
        files = list(Path(self.tmp.name).glob("*.sqac"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stem, "memory")

    def test_split_creates_shard(self):
        """Writing past the threshold creates a new shard file."""
        rack = CartridgeRack(
            directory=self.tmp.name,
            default="memory",
            max_entries=5,
            auto_load=False,
        )
        rack.create("memory")
        # Fill to threshold
        for i in range(5):
            rack.write("memory", f"fact {i}", key=f"k{i}", kind="fact")
        # This write triggers a split
        rack.write("memory", "overflow fact", key="overflow", kind="fact")
        rack.save()
        # Should have 2 files: memory.sqac + memory__2.sqac
        files = sorted(p.stem for p in Path(self.tmp.name).glob("*.sqac"))
        self.assertIn("memory", files)
        self.assertIn("memory__2", files)
        self.assertEqual(len(files), 2)

    def test_shard_naming_increments(self):
        """Multiple splits create __2, __3, etc."""
        rack = CartridgeRack(
            directory=self.tmp.name,
            default="memory",
            max_entries=3,
            auto_load=False,
        )
        rack.create("memory")
        # Fill to first threshold
        for i in range(3):
            rack.write("memory", f"fact {i}", key=f"k{i}", kind="fact")
        # Trigger split 1
        rack.write("memory", "overflow 1", key="o1", kind="fact")
        # Fill second shard
        for i in range(3):
            rack.write("memory", f"fact b{i}", key=f"kb{i}", kind="fact")
        # Trigger split 2
        rack.write("memory", "overflow 2", key="o2", kind="fact")
        rack.save()
        files = sorted(p.stem for p in Path(self.tmp.name).glob("*.sqac"))
        self.assertIn("memory", files)
        self.assertIn("memory__2", files)
        self.assertIn("memory__3", files)
        self.assertEqual(len(files), 3)

    def test_search_merges_across_shards(self):
        """Search finds results across all shards."""
        rack = CartridgeRack(
            directory=self.tmp.name,
            default="memory",
            max_entries=3,
            auto_load=False,
        )
        rack.create("memory")
        # Write to first shard
        rack.write("memory", "ARM64 deploy rule", key="deploy", kind="fact")
        rack.write("memory", "Stripe payment processor", key="payment", kind="fact")
        rack.write("memory", "Error budget 0.1%", key="slo", kind="fact")
        # Trigger split
        rack.write("memory", "Vault stores secrets", key="vault", kind="fact")
        rack.save()
        # Search should find results from both shards
        hits = rack.search("deploy", top_k=5)
        contents = [h.content for h in hits]
        self.assertTrue(
            any("ARM64" in c for c in contents),
            f"Expected ARM64 in results, got {contents}"
        )

    def test_search_deduplicates_across_shards(self):
        """Same content in multiple shards is deduplicated."""
        rack = CartridgeRack(
            directory=self.tmp.name,
            default="memory",
            max_entries=3,
            auto_load=False,
        )
        rack.create("memory")
        rack.write("memory", "fact one", key="k1", kind="fact")
        rack.write("memory", "fact two", key="k2", kind="fact")
        rack.write("memory", "fact three", key="k3", kind="fact")
        # Trigger split
        rack.write("memory", "fact four", key="k4", kind="fact")
        rack.save()
        # Search for "fact" should not return duplicates
        hits = rack.search("fact", top_k=10)
        contents = [h.content for h in hits]
        self.assertEqual(len(contents), len(set(contents)))

    def test_split_with_reload(self):
        """Shards are discovered on reload."""
        # First session: create with split
        rack = CartridgeRack(
            directory=self.tmp.name,
            default="memory",
            max_entries=3,
            auto_load=False,
        )
        rack.create("memory")
        for i in range(4):
            rack.write("memory", f"fact {i}", key=f"k{i}", kind="fact")
        rack.save()
        # Second session: reload discovers shards
        rack2 = CartridgeRack(
            directory=self.tmp.name,
            default="memory",
            max_entries=3,
            auto_load=True,
        )
        # Should find all 4 entries across shards
        hits = rack2.search("fact", top_k=10)
        self.assertEqual(len(hits), 4)


class TestFolderRouting(unittest.TestCase):
    """Folder routing organizes cartridges into kind-based subdirectories."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_write_to_kind_folder(self):
        """Writes with kind routing go to the correct subfolder."""
        rack = CartridgeRack(
            directory=self.tmp.name,
            routes={"fact": "facts", "skill": "skills"},
            default="facts",
            folder_routing=True,
            auto_load=False,
        )
        rack.write_routed("ARM64 deploy rule", kind="fact")
        rack.write_routed("weighted-index skill", kind="skill")
        rack.save()
        # Check directory structure
        facts_dir = Path(self.tmp.name) / "facts"
        skills_dir = Path(self.tmp.name) / "skills"
        self.assertTrue(facts_dir.exists())
        self.assertTrue(skills_dir.exists())
        # Files should be in the right folders
        fact_files = list(facts_dir.glob("*.sqac"))
        skill_files = list(skills_dir.glob("*.sqac"))
        self.assertEqual(len(fact_files), 1)
        self.assertEqual(len(skill_files), 1)

    def test_auto_load_recurses(self):
        """Auto-load discovers cartridges in subdirectories."""
        # Create folder structure manually
        facts_dir = Path(self.tmp.name) / "facts"
        skills_dir = Path(self.tmp.name) / "skills"
        facts_dir.mkdir()
        skills_dir.mkdir()
        # Create cartridges in subdirs
        store1 = SqacStore()
        store1.add("fact one", key="k1", kind="fact")
        store1.save(facts_dir / "team.sqac")
        store2 = SqacStore()
        store2.add("skill one", key="k2", kind="skill")
        store2.save(skills_dir / "logic.sqac")
        # Rack should discover both
        rack = CartridgeRack(
            directory=self.tmp.name,
            auto_load=True,
        )
        names = rack.names()
        # Names use __ separator for subdirectory paths
        self.assertTrue(
            any("facts" in n for n in names),
            f"Expected facts cartridge in {names}"
        )
        self.assertTrue(
            any("skills" in n for n in names),
            f"Expected skills cartridge in {names}"
        )

    def test_auto_split_in_folder(self):
        """Auto-split creates shards within the same folder."""
        rack = CartridgeRack(
            directory=self.tmp.name,
            routes={"fact": "facts"},
            default="facts",
            max_entries=3,
            folder_routing=True,
            auto_load=False,
        )
        # Fill to threshold
        for i in range(3):
            rack.write_routed(f"fact {i}", key=f"k{i}", kind="fact")
        # Trigger split
        rack.write_routed("overflow fact", key="overflow", kind="fact")
        rack.save()
        # Both files should be in facts/
        facts_dir = Path(self.tmp.name) / "facts"
        files = sorted(p.stem for p in facts_dir.glob("*.sqac"))
        self.assertTrue(len(files) >= 2)
        # First file is the base, second is the shard
        self.assertTrue(
            any("facts" in f for f in files),
            f"Expected facts base in {files}"
        )
        self.assertTrue(
            any("__" in f for f in files),
            f"Expected shard in {files}"
        )

    def test_folder_backward_compat(self):
        """Flat files (no folders) still work when folder_routing=False."""
        rack = CartridgeRack(
            directory=self.tmp.name,
            default="memory",
            folder_routing=False,
            auto_load=False,
        )
        rack.write_routed("fact one", kind="fact")
        rack.save()
        # File should be at root level
        root_files = list(Path(self.tmp.name).glob("*.sqac"))
        self.assertEqual(len(root_files), 1)
        self.assertEqual(root_files[0].stem, "memory")


class TestAutoSplitIntegration(unittest.TestCase):
    """Integration tests combining auto-split with existing features."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_graduation_into_sharded_cartridge(self):
        """Graduation works even when the target is sharded."""
        from sqac.offloader import ContextOffloader

        rack = CartridgeRack(
            directory=self.tmp.name,
            default="facts",
            max_entries=3,
            auto_load=False,
        )
        rack.create("facts")
        # Fill facts to threshold
        for i in range(3):
            rack.write("facts", f"existing fact {i}", key=f"ef{i}", kind="fact")
        # Create a session with high-salience content
        off = ContextOffloader(
            Path(self.tmp.name) / "session.sqac", window=8, semantic=False
        )
        off.observe("user", "CRITICAL: Our deploy pipeline uses ARM64 runners only, no AMD64 allowed. Error: deployment target mismatch.")
        off.observe("assistant", "ERROR confirmed: ARM64 only for deployments. AMD64 images are NOT supported. Fix: update CI matrix to ARM64.")
        off.save()  # flush exchanges to _store
        # Graduate with low threshold so even modest salience promotes
        result = rack.graduate(off, target_name="facts", threshold=0.01)
        self.assertGreater(result["promoted"], 0)


if __name__ == "__main__":
    unittest.main()
