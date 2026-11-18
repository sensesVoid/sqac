#!/usr/bin/env python3
"""Tests for sqac.integrations — OpenAI-compatible memory middleware."""
import tempfile
import unittest
from pathlib import Path

from sqac.integrations import wrap


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = []
        self.response = None

    def create(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        return self.response


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    """Minimal OpenAI-compatible client (Ollama / vLLM / LiteLLM shape)."""

    def __init__(self, reply="hello"):
        self.chat = _FakeChat()
        self.chat.completions.response = _Response(reply)
        self.models = "UPSTREAM-MODELS"


class TestMemoryWrap(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)
        self.client = _FakeClient("deploy answer")
        self.wrapped = wrap(self.client, memory_dir=self.d,
                            threshold=0.0, top_k=3)  # threshold 0: any hit counts
        self.wrapped.rack.write_routed("deploy ARM64 first", key="deploy-arch",
                                       source="test", kind="fact")

    def tearDown(self):
        self._td.cleanup()

    def _called_messages(self):
        return self.client.chat.completions.calls[0][1]

    def test_memory_is_hydrated_into_system_prompt(self):
        self.wrapped.chat.completions.create(
            model="llama3.1",
            messages=[{"role": "user", "content": "tell me about deploy ARM64 first please"}],
        )
        msgs = self._called_messages()
        sys = next(m for m in msgs if m["role"] == "system")
        self.assertIn("Relevant memories", sys["content"])
        self.assertIn("deploy ARM64 first", sys["content"])

    def test_existing_system_message_is_extended_not_replaced(self):
        self.wrapped.chat.completions.create(
            model="llama3.1",
            messages=[{"role": "system", "content": "You are terse."},
                      {"role": "user", "content": "tell me about deploy ARM64 first please"}],
        )
        sys = next(m for m in self._called_messages() if m["role"] == "system")
        self.assertIn("You are terse.", sys["content"])
        self.assertIn("deploy ARM64 first", sys["content"])

    def test_fall_through_to_upstream_attributes(self):
        self.assertEqual(self.wrapped.models, "UPSTREAM-MODELS")

    def test_response_passes_through_untouched(self):
        resp = self.wrapped.chat.completions.create(
            model="llama3.1",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIs(resp, self.client.chat.completions.response)

    def test_turns_are_recorded_offloader(self):
        self.wrapped.chat.completions.create(
            model="llama3.1",
            messages=[{"role": "user", "content": "can we ship x86 images?"}],
        )
        self.assertEqual(self.wrapped.offloader._turn_count, 2)
        self.assertEqual(self.wrapped.offloader._since_flush, 2)

    def test_no_hits_means_no_injection(self):
        fresh_dir = self.d / "empty-mem"
        wrapped = wrap(_FakeClient("answer"), memory_dir=fresh_dir,
                       threshold=0.99)
        wrapped.chat.completions.create(
            model="m",
            messages=[{"role": "user", "content": "unrelated topic about cooking"}],
        )
        msgs = wrapped._upstream.chat.completions.calls[0][1]
        self.assertFalse(any(m.get("role") == "system" and "Relevant memories" in (m.get("content") or "")
                             for m in msgs))


if __name__ == "__main__":
    unittest.main()
