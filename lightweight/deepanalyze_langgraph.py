import contextlib
from datetime import datetime
import io
import json
import logging
import os
import re
import time
import traceback
import uuid
from typing import Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI


class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    response_chunks: List[str]
    round_idx: int
    max_rounds: int
    pending_code: Optional[str]
    finished: bool
    temperature: float
    max_tokens: int
    top_p: Optional[float]
    top_k: Optional[int]
    plan_text: str
    final_answer: str
    last_execution_output: str
    consecutive_exec_failures: int
    max_exec_retries: int


class DeepAnalyzeLangGraph:
    """
    基于 LangGraph 的节点化轻量 Agent：
    Planner -> Coder -> Executor -> Reporter。
    使用单一 OpenAI 兼容 API 通道（`api_base` 可选）。
    """

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_rounds: int = 30,
        request_timeout: int = 120,
        max_api_retries: int = 3,
        max_exec_retries: int = 2,
        log_level: str = "INFO",
    ):
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.request_timeout = request_timeout
        self.max_api_retries = max_api_retries
        self.max_exec_retries = max_exec_retries
        self.run_id = uuid.uuid4().hex[:12]
        self.logger = self._build_logger(log_level)
        self.client = self._build_client(api_base=api_base, api_key=api_key)
        self.graph = self._build_graph()

    def _build_logger(self, log_level: str) -> logging.Logger:
        logger_name = f"DeepAnalyzeLangGraph.{self.run_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    def _log_event(self, node: str, event: str, **fields: object) -> None:
        payload: Dict[str, object] = {"run_id": self.run_id, "node": node, "event": event}
        payload.update(fields)
        self.logger.info(json.dumps(payload, ensure_ascii=False))

    def _build_client(self, api_base: Optional[str], api_key: Optional[str]) -> OpenAI:
        key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("API key is required. Set API_KEY or OPENAI_API_KEY.")
        base = api_base or os.getenv("API_BASE")
        if base:
            return OpenAI(base_url=base, api_key=key, timeout=self.request_timeout)
        return OpenAI(api_key=key, timeout=self.request_timeout)

    def execute_code(self, code_str: str) -> str:
        """执行 Python 代码，返回标准输出/错误输出，并格式化异常信息。"""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                stderr_capture
            ):
                exec(code_str, {})
            output = stdout_capture.getvalue()
            if stderr_capture.getvalue():
                output += stderr_capture.getvalue()
            return output
        except Exception as exec_error:
            code_lines = code_str.splitlines()
            tb_lines = traceback.format_exc().splitlines()
            error_line = None

            for line in tb_lines:
                if 'File "<string>", line' in line:
                    try:
                        line_num = int(line.split(", line ")[1].split(",")[0])
                        error_line = line_num
                        break
                    except (IndexError, ValueError):
                        continue

            error_message = "Traceback (most recent call last):\n"
            if error_line and 1 <= error_line <= len(code_lines):
                error_message += f'  File "<string>", line {error_line}, in <module>\n'
                error_message += f"    {code_lines[error_line - 1].strip()}\n"
            error_message += f"{type(exec_error).__name__}: {str(exec_error)}"
            if stderr_capture.getvalue():
                error_message += f"\n{stderr_capture.getvalue()}"
            return f"[Error]:\n{error_message.strip()}"

    def _chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: Optional[float],
        top_k: Optional[int],
        stop: Optional[List[str]],
        node: str,
    ) -> str:
        api_messages = self._normalize_messages_for_api(messages)
        kwargs: Dict[str, object] = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["extra_body"] = {"top_k": top_k, "add_generation_prompt": False}

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_api_retries + 1):
            try:
                self._log_event(node, "llm_call", attempt=attempt)
                completion = self.client.chat.completions.create(**kwargs)
                answer = completion.choices[0].message.content or ""
                return answer
            except Exception as err:  # noqa: BLE001
                last_error = err
                self._log_event(node, "llm_call_failed", attempt=attempt, error=str(err))
                if attempt < self.max_api_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(f"LLM call failed after retries: {last_error}") from last_error

    @staticmethod
    def _normalize_messages_for_api(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        将内部消息角色标准化为 OpenAI 兼容接口可接受的角色集合。
        允许角色：system / user / assistant / tool / function。
        """
        allowed_roles = {"system", "user", "assistant", "tool", "function"}
        normalized: List[Dict[str, str]] = []

        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            if role in allowed_roles:
                normalized.append({"role": role, "content": content})
                continue

            # 内部 execute 角色转换为 user，避免接口 400，并保留语义
            if role == "execute":
                normalized.append(
                    {
                        "role": "user",
                        "content": f"以下是代码执行结果，请据此继续：\n<Execute>\n{content}\n</Execute>",
                    }
                )
                continue

            # 兜底：未知角色统一按 user 发送
            normalized.append({"role": "user", "content": content})

        return normalized

    @staticmethod
    def _extract_code(ans: str) -> Optional[str]:
        code_match = re.search(r"<Code>(.*?)</Code>", ans, re.DOTALL)
        if not code_match:
            return None
        code_content = code_match.group(1).strip()
        md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
        return md_match.group(1).strip() if md_match else code_content

    @staticmethod
    def _extract_answer(ans: str) -> str:
        answer_match = re.search(r"<Answer>(.*?)</Answer>", ans, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return ""

    def _planner_node(self, state: AgentState) -> AgentState:
        if state["finished"] or state["round_idx"] >= state["max_rounds"]:
            state["finished"] = True
            return state

        user_prompt = state["messages"][0]["content"]
        planning_context = (
            f"用户任务：\n{user_prompt}\n\n"
            f"上一轮执行输出：\n{state['last_execution_output'] or '（无）'}\n\n"
            f"当前轮次：{state['round_idx']} / {state['max_rounds']}"
        )
        planning_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是规划节点。请为数据分析代码代理生成精炼、可执行的计划。"
                    "只输出 2-5 条简短步骤，不要输出代码。"
                ),
            },
            {"role": "user", "content": planning_context},
        ]
        plan = self._chat_with_retry(
            planning_messages,
            state["temperature"],
            min(state["max_tokens"], 512),
            state["top_p"],
            state["top_k"],
            stop=None,
            node="planner",
        ).strip()
        state["plan_text"] = plan
        state["response_chunks"].append(f"<Plan>\n{plan}\n</Plan>")
        self._log_event("planner", "plan_generated", round_idx=state["round_idx"])
        return state

    def _coder_node(self, state: AgentState) -> AgentState:
        if state["finished"] or state["round_idx"] >= state["max_rounds"]:
            state["finished"] = True
            return state

        coder_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是数据科学代理中的编码节点。请严格遵循 DeepAnalyze 标签协议："
                    "需要推理时使用 <Analyze>...</Analyze>，需要执行代码时仅在 "
                    "<Code>...</Code> 中输出可执行 Python，最终完成时使用 "
                    "<Answer>...</Answer> 给出结论。"
                ),
            },
            {"role": "system", "content": f"当前计划：\n{state['plan_text']}"},
        ]
        if state["last_execution_output"]:
            coder_messages.append(
                {
                    "role": "system",
                    "content": f"上一轮执行输出：\n{state['last_execution_output']}",
                }
            )
        coder_messages.extend(state["messages"])

        ans = self._chat_with_retry(
            coder_messages,
            state["temperature"],
            state["max_tokens"],
            state["top_p"],
            state["top_k"],
            stop=["</Code>"],
            node="coder",
        )
        if ans.count("<Code>") > ans.count("</Code>"):
            ans += "</Code>"

        state["response_chunks"].append(ans)
        state["messages"].append({"role": "assistant", "content": ans})
        state["round_idx"] += 1
        state["pending_code"] = self._extract_code(ans)

        answer_text = self._extract_answer(ans)
        if answer_text:
            state["final_answer"] = answer_text
            state["finished"] = True
            state["pending_code"] = None

        self._log_event(
            "coder",
            "coder_output",
            round_idx=state["round_idx"],
            has_code=bool(state["pending_code"]),
            has_answer=bool(answer_text),
        )
        return state

    def _executor_node(self, state: AgentState) -> AgentState:
        code_str = state.get("pending_code")
        if not code_str:
            return state

        self._log_event("executor", "execution_start", round_idx=state["round_idx"])
        exe_output = self.execute_code(code_str)
        state["last_execution_output"] = exe_output
        state["response_chunks"].append(f"<Execute>\n{exe_output}\n</Execute>")
        state["messages"].append({"role": "execute", "content": exe_output})
        state["pending_code"] = None

        if exe_output.startswith("[Error]:") or exe_output.startswith("[Timeout]:"):
            state["consecutive_exec_failures"] += 1
            self._log_event(
                "executor",
                "execution_failed",
                failures=state["consecutive_exec_failures"],
                max_retries=state["max_exec_retries"],
            )
        else:
            state["consecutive_exec_failures"] = 0
            self._log_event("executor", "execution_success")
        return state

    def _reporter_node(self, state: AgentState) -> AgentState:
        if state["final_answer"]:
            report_text = f"<Answer>\n{state['final_answer']}\n</Answer>"
            state["response_chunks"].append(report_text)
            self._log_event("reporter", "used_existing_answer")
            state["finished"] = True
            return state

        reporter_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是报告节点。请输出简洁最终结论，并使用 <Answer>...</Answer> 包裹。"
                    "若任务未完成，请在结论中明确说明剩余阻塞项。"
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(state["response_chunks"][-20:]),
            },
        ]
        report = self._chat_with_retry(
            reporter_messages,
            state["temperature"],
            min(state["max_tokens"], 1024),
            state["top_p"],
            state["top_k"],
            stop=None,
            node="reporter",
        ).strip()
        if "<Answer>" not in report:
            report = f"<Answer>\n{report}\n</Answer>"
        state["response_chunks"].append(report)
        state["final_answer"] = self._extract_answer(report)
        state["finished"] = True
        self._log_event("reporter", "report_generated")
        return state

    @staticmethod
    def _route_after_coder(state: AgentState) -> str:
        if state.get("finished") or state["round_idx"] >= state["max_rounds"]:
            return "reporter"
        if state.get("pending_code"):
            return "executor"
        return "planner"

    @staticmethod
    def _route_after_executor(state: AgentState) -> str:
        if state.get("finished") or state["round_idx"] >= state["max_rounds"]:
            return "reporter"
        if state["consecutive_exec_failures"] > state["max_exec_retries"]:
            return "reporter"
        if state["consecutive_exec_failures"] > 0:
            return "coder"
        return "planner"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("coder", self._coder_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("reporter", self._reporter_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "coder")
        graph.add_conditional_edges(
            "coder",
            self._route_after_coder,
            {"executor": "executor", "planner": "planner", "reporter": "reporter"},
        )
        graph.add_conditional_edges(
            "executor",
            self._route_after_executor,
            {"coder": "coder", "planner": "planner", "reporter": "reporter"},
        )
        graph.add_edge("reporter", END)
        return graph.compile()

    def _save_report_markdown(self, workspace: str, content: str) -> str:
        report_dir = os.path.join(workspace, "reports")
        os.makedirs(report_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{ts}_{self.run_id}.md"
        report_path = os.path.abspath(os.path.join(report_dir, filename))
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        self._log_event("reporter", "report_saved", report_path=report_path)
        return report_path

    def generate(
        self,
        prompt: str,
        workspace: str,
        temperature: float = 0.5,
        max_tokens: int = 32768,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        执行 LangGraph 工作流并返回完整推理轨迹。
        """
        original_cwd = os.getcwd()
        os.makedirs(workspace, exist_ok=True)
        os.chdir(workspace)
        self._log_event("runtime", "run_start")

        state: AgentState = {
            "messages": [{"role": "user", "content": prompt}],
            "response_chunks": [],
            "round_idx": 0,
            "max_rounds": self.max_rounds,
            "pending_code": None,
            "finished": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "plan_text": "",
            "final_answer": "",
            "last_execution_output": "",
            "consecutive_exec_failures": 0,
            "max_exec_retries": self.max_exec_retries,
        }

        try:
            final_state = self.graph.invoke(
                state,
                config={"recursion_limit": max(200, self.max_rounds * 8)},
            )
            reasoning = "\n".join(final_state["response_chunks"])
            report_body = final_state.get("final_answer", "").strip()
            if not report_body:
                report_body = self._extract_answer(reasoning).strip() or reasoning
            report_md = f"# 分析报告\n\n{report_body}\n"
            report_path = self._save_report_markdown(workspace, report_md)
            self._log_event(
                "runtime",
                "run_end",
                rounds=final_state["round_idx"],
                finished=final_state["finished"],
                report_path=report_path,
            )
        except Exception as err:  # noqa: BLE001
            self._log_event("runtime", "run_crashed", error=str(err))
            reasoning = "\n".join(state["response_chunks"])
            report_md = f"# 分析报告（异常中断）\n\n{reasoning}\n"
            report_path = self._save_report_markdown(workspace, report_md)
        finally:
            os.chdir(original_cwd)

        return {"reasoning": reasoning, "report_path": report_path}


# 向后兼容别名
DeepAnalyzeVLLM = DeepAnalyzeLangGraph
