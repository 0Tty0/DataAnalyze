import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepanalyze_langgraph import DeepAnalyzeLangGraph


class ContextCompressionTests(unittest.TestCase):
    def _build_stub(self) -> DeepAnalyzeLangGraph:
        # 跳过 __init__，只测试上下文压缩逻辑。
        agent = DeepAnalyzeLangGraph.__new__(DeepAnalyzeLangGraph)
        agent.max_context_tokens = 80
        agent.reporter_context_tokens = 20
        agent._log_event = lambda *args, **kwargs: None
        return agent

    def test_reporter_context_uses_token_budget(self) -> None:
        agent = self._build_stub()
        chunks = [
            "<Analyze>" + ("A" * 120) + "</Analyze>",
            "<Code>print('x')</Code>",
            "<Execute>ok</Execute>",
            "<Answer>简短结论</Answer>",
        ]
        text = agent._build_reporter_context(chunks)
        self.assertIn("<Answer>", text)
        self.assertNotIn("A" * 40, text)

    def test_coder_context_compression_reduces_message_count(self) -> None:
        agent = self._build_stub()
        agent.max_context_tokens = 120
        agent._build_middle_summary = lambda middle, state: "中段摘要"

        long_text = "数据分析步骤" * 60
        history = [{"role": "user", "content": f"第{i}轮: {long_text}"} for i in range(12)]

        state = {
            "messages": history,
            "temperature": 0.5,
            "max_tokens": 1024,
            "top_p": None,
            "top_k": None,
        }
        compressed = agent._compress_messages_for_coder(state)
        self.assertLess(len(compressed), len(history))
        self.assertTrue(any(str(m.get("content", "")).startswith("[") for m in compressed))


if __name__ == "__main__":
    unittest.main()
