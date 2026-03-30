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


class AsyncGraphInvokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_agenerate_uses_graph_ainvoke(self) -> None:
        agent = DeepAnalyzeLangGraph.__new__(DeepAnalyzeLangGraph)
        agent.max_rounds = 3
        agent.max_exec_retries = 2
        agent._log_event = lambda *args, **kwargs: None
        agent._sanitize_report_text = lambda text: text
        agent._extract_answer = lambda text: "????"
        agent._save_report_markdown = lambda workspace, content: str(Path(workspace) / "report.md")

        class FakeGraph:
            def __init__(self) -> None:
                self.called = False

            async def ainvoke(self, state, config=None):
                self.called = True
                return {
                    **state,
                    "response_chunks": ["<Answer>????</Answer>"],
                    "final_answer": "????",
                    "round_idx": 1,
                    "finished": True,
                }

        agent.graph = FakeGraph()
        workspace = Path(__file__).resolve().parent / "_tmp_async_graph"
        workspace.mkdir(parents=True, exist_ok=True)
        result = await DeepAnalyzeLangGraph.agenerate(agent, "????", str(workspace))
        self.assertTrue(agent.graph.called)
        self.assertIn("reasoning", result)
        self.assertIn("report_path", result)
