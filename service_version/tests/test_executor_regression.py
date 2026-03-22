import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepanalyze_langgraph import DeepAnalyzeLangGraph

_TEST_TMP_ROOT = Path(__file__).resolve().parent / "_tmp"
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _build_agent(exec_timeout_sec: int = 1) -> DeepAnalyzeLangGraph:
    # 仅验证执行器分支，不触发真实 LLM 请求。
    return DeepAnalyzeLangGraph(
        model_name="gpt-4.1-mini",
        api_key="dummy",
        max_rounds=1,
        exec_timeout_sec=exec_timeout_sec,
        exec_memory_mb=256,
        exec_cpu_seconds=2,
    )


class ExecutorRegressionTests(unittest.TestCase):
    def test_execute_code_timeout_branch(self) -> None:
        agent = _build_agent(exec_timeout_sec=1)
        with tempfile.TemporaryDirectory(dir=_TEST_TMP_ROOT) as workspace:
            out = agent.execute_code("import time\ntime.sleep(2)", workspace)
        self.assertTrue(out.startswith("[Timeout]:"), out)

    def test_execute_code_error_branch(self) -> None:
        agent = _build_agent(exec_timeout_sec=2)
        with tempfile.TemporaryDirectory(dir=_TEST_TMP_ROOT) as workspace:
            out = agent.execute_code("raise ValueError('boom')", workspace)
        self.assertTrue(out.startswith("[Error]:"), out)
        self.assertIn("ValueError", out)


if __name__ == "__main__":
    unittest.main()
