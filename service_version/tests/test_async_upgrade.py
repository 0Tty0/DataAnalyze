import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepanalyze_langgraph import DeepAnalyzeLangGraph
from service.app import chat_completions


class AsyncUpgradeTests(unittest.TestCase):
    def test_agent_exposes_async_entry(self) -> None:
        self.assertTrue(hasattr(DeepAnalyzeLangGraph, "agenerate"))
        self.assertTrue(inspect.iscoroutinefunction(DeepAnalyzeLangGraph.agenerate))

    def test_chat_endpoint_is_async(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(chat_completions))


if __name__ == "__main__":
    unittest.main()
