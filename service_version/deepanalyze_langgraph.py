from datetime import datetime
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI, AsyncOpenAI


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
    workspace: str


class DeepAnalyzeLangGraph:
    """基于 LangGraph 的节点化数据分析代理：Planner -> Coder -> Executor -> Reporter。"""

    def __init__(
        self,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_rounds: int = 30,
        request_timeout: int = 120,
        max_api_retries: int = 3,
        max_exec_retries: int = 2,
        exec_timeout_sec: int = 30,
        exec_memory_mb: int = 512,
        exec_cpu_seconds: int = 30,
        max_context_tokens: int = 8000,
        reporter_context_tokens: int = 3000,
        log_level: str = "INFO",
    ):
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.request_timeout = request_timeout
        self.max_api_retries = max_api_retries
        self.max_exec_retries = max_exec_retries
        self.exec_timeout_sec = exec_timeout_sec
        self.exec_memory_mb = exec_memory_mb
        self.exec_cpu_seconds = exec_cpu_seconds
        self.max_context_tokens = max(int(max_context_tokens), 1024)
        self.reporter_context_tokens = max(int(reporter_context_tokens), 512)
        self.run_id = uuid.uuid4().hex[:12]
        self.logger = self._build_logger(log_level)
        self.client = self._build_client(api_base=api_base, api_key=api_key)
        self.async_client = self._build_async_client(api_base=api_base, api_key=api_key)
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
            raise ValueError("需要提供 API Key，请设置 API_KEY 或 OPENAI_API_KEY。")
        base = api_base or os.getenv("API_BASE")
        if base:
            return OpenAI(base_url=base, api_key=key, timeout=self.request_timeout)
        return OpenAI(api_key=key, timeout=self.request_timeout)

    def _build_async_client(self, api_base: Optional[str], api_key: Optional[str]) -> AsyncOpenAI:
        key = api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("???? API Key???? API_KEY ? OPENAI_API_KEY?")
        base = api_base or os.getenv("API_BASE")
        if base:
            return AsyncOpenAI(base_url=base, api_key=key, timeout=self.request_timeout)
        return AsyncOpenAI(api_key=key, timeout=self.request_timeout)

    def _subprocess_limits(self) -> Dict[str, Any]:
        """? POSIX ????????????Windows ??????"""
        kwargs: Dict[str, Any] = {}
        if os.name != "posix":
            return kwargs

        try:
            import resource
        except Exception:  # noqa: BLE001
            return kwargs

        memory_bytes = max(int(self.exec_memory_mb), 64) * 1024 * 1024
        cpu_seconds = max(int(self.exec_cpu_seconds), 1)

        def _limit_resources() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        kwargs["preexec_fn"] = _limit_resources
        return kwargs

    def execute_code(self, code_str: str, workspace: str) -> str:
        """?????????????? stdout/stderr?"""
        os.makedirs(workspace, exist_ok=True)
        tmp_path = ""

        try:
            fd, tmp_path = tempfile.mkstemp(prefix="agent_exec_", suffix=".py", dir=workspace)
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code_str)

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            run_kwargs: Dict[str, Any] = {
                "args": [sys.executable, tmp_path],
                "cwd": workspace,
                "capture_output": True,
                "text": True,
                "timeout": self.exec_timeout_sec,
                "env": env,
            }
            run_kwargs.update(self._subprocess_limits())
            completed = subprocess.run(**run_kwargs)

            output = (completed.stdout or "") + (completed.stderr or "")
            if completed.returncode != 0:
                err_text = output.strip() or f"Process exited with code {completed.returncode}"
                return f"[Error]:\n{err_text}"
            return output
        except subprocess.TimeoutExpired as timeout_err:
            timeout_output = (timeout_err.stdout or "") + (timeout_err.stderr or "")
            return f"[Timeout]: execution exceeded {self.exec_timeout_sec} seconds\n{timeout_output}".strip()
        except Exception as exec_error:  # noqa: BLE001
            return f"[Error]:\n{type(exec_error).__name__}: {exec_error}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


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
                return completion.choices[0].message.content or ""
            except Exception as err:  # noqa: BLE001
                last_error = err
                self._log_event(node, "llm_call_failed", attempt=attempt, error=str(err))
                if attempt < self.max_api_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(f"LLM call failed after retries: {last_error}") from last_error

    async def _achat_with_retry(
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
                self._log_event(node, "llm_call_async", attempt=attempt)
                completion = await self.async_client.chat.completions.create(**kwargs)
                return completion.choices[0].message.content or ""
            except Exception as err:  # noqa: BLE001
                last_error = err
                self._log_event(node, "llm_call_async_failed", attempt=attempt, error=str(err))
                if attempt < self.max_api_retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(f"Async LLM call failed after retries: {last_error}") from last_error

    @staticmethod
    def _normalize_messages_for_api(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """把内部消息规范化为 OpenAI 兼容格式。"""
        allowed_roles = {"system", "user", "assistant", "tool", "function"}
        normalized: List[Dict[str, str]] = []

        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            if role in allowed_roles:
                normalized.append({"role": role, "content": content})
                continue

            if role == "execute":
                normalized.append(
                    {
                        "role": "user",
                        "content": f"以下是代码执行结果，请据此继续分析和修正：\n<Execute>\n{content}\n</Execute>",
                    }
                )
                continue

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

    @staticmethod
    def _sanitize_report_text(text: str) -> str:
        lines = []
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            if any(keyword in stripped for keyword in ["已保存到", "保存到", "下载链接", "文件路径", "导出完成", "见下方"]):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        return cleaned or (text or "").strip()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # ?????????????? 4 ??? 1 token?
        return max(1, len(str(text or "")) // 4)

    def _estimate_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += self._estimate_tokens(msg.get("role", ""))
            total += self._estimate_tokens(msg.get("content", ""))
        return total

    def _build_middle_summary(self, middle_messages: List[Dict[str, str]], state: AgentState) -> str:
        if not middle_messages:
            return ""

        merged_parts: List[str] = []
        for msg in middle_messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            merged_parts.append(f"[{role}]\n{content}")

        if not merged_parts:
            return ""

        raw_text = "\n\n".join(merged_parts)
        if self._estimate_tokens(raw_text) <= 1200:
            return raw_text

        summary_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "??????????????????????????????"
                    "??????????????????????????"
                    "??????????????"
                ),
            },
            {"role": "user", "content": raw_text},
        ]

        try:
            summary = self._chat_with_retry(
                summary_messages,
                state["temperature"],
                min(512, state["max_tokens"]),
                state["top_p"],
                state["top_k"],
                stop=None,
                node="context",
            ).strip()
            return summary or raw_text[:4000]
        except Exception as err:  # noqa: BLE001
            self._log_event("context", "summary_failed", error=str(err))
            return raw_text[:4000]

    def _compress_messages_for_coder(self, state: AgentState) -> List[Dict[str, str]]:
        history = list(state["messages"])
        budget = self.max_context_tokens

        total_tokens = self._estimate_messages_tokens(history)
        if total_tokens <= budget:
            return history

        system_msgs = [m for m in history if m.get("role") == "system"]
        non_system = [m for m in history if m.get("role") != "system"]

        if len(non_system) <= 8:
            return history[-8:]

        anchor = non_system[:1]
        recent_keep = 8
        middle = non_system[1:-recent_keep]
        recent = non_system[-recent_keep:]

        middle_summary = self._build_middle_summary(middle, state)

        compressed: List[Dict[str, str]] = []
        if system_msgs:
            compressed.extend(system_msgs[-2:])
        compressed.extend(anchor)
        if middle_summary:
            compressed.append({"role": "user", "content": f"[????]\n{middle_summary}"})
        compressed.extend(recent)

        while self._estimate_messages_tokens(compressed) > budget and len(recent) > 2:
            recent = recent[1:]
            compressed = []
            if system_msgs:
                compressed.extend(system_msgs[-2:])
            compressed.extend(anchor)
            if middle_summary:
                compressed.append({"role": "user", "content": f"[????]\n{middle_summary}"})
            compressed.extend(recent)

        self._log_event(
            "context",
            "compressed",
            before_tokens=total_tokens,
            after_tokens=self._estimate_messages_tokens(compressed),
            before_messages=len(history),
            after_messages=len(compressed),
        )
        return compressed

    def _build_reporter_context(self, chunks: List[str]) -> str:
        if not chunks:
            return ""

        budget = self.reporter_context_tokens
        picked: List[str] = []
        used = 0

        for chunk in reversed(chunks):
            tks = self._estimate_tokens(chunk)
            if picked and (used + tks > budget):
                break
            picked.append(chunk)
            used += tks

        picked.reverse()
        context_text = "\n\n".join(picked)
        self._log_event("context", "reporter_context_built", chunks=len(picked), tokens=used)
        return context_text

    def _planner_node(self, state: AgentState) -> AgentState:
        if state["finished"] or state["round_idx"] >= state["max_rounds"]:
            state["finished"] = True
            return state

        user_prompt = state["messages"][0]["content"]
        planning_context = (
            f"当前任务：\n{user_prompt}\n\n"
            f"上一轮执行结果：\n{state['last_execution_output'] or '无'}\n\n"
            f"当前轮次：{state['round_idx']} / {state['max_rounds']}"
        )
        planning_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "你是规划节点。请用简洁中文概括当前任务的分析路线，指出要检查的数据、关键计算步骤和输出重点。不要写代码，不要给最终答案。",
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
                    "你是编码节点。请严格遵循 DeepAnalyze 标签协议：需要思考时使用 <Analyze>...</Analyze>，"
                    "需要执行代码时仅在 <Code>...</Code> 中输出可执行 Python，最终必须在 <Answer>...</Answer> 中给出完整的中文分析报告正文。"
                    "<Answer> 必须是最终报告本体，不要写“分析内容已保存到 xx”、不要写文件路径、不要写下载说明、不要写多余寒暄。"
                ),
            },
            {"role": "system", "content": f"当前计划：\n{state['plan_text']}"},
        ]
        if state["last_execution_output"]:
            coder_messages.append({"role": "system", "content": f"上一轮执行输出：\n{state['last_execution_output']}"})
        compressed_history = self._compress_messages_for_coder(state)
        coder_messages.extend(compressed_history)

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
            state["final_answer"] = self._sanitize_report_text(answer_text)
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
        exe_output = self.execute_code(code_str, state["workspace"])
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
                    "你是报告节点。请把前面所有分析、代码执行结果和结论整理为一份完整的中文分析报告。"
                    "最终必须使用 <Answer>...</Answer> 包裹报告正文，报告中不要出现“已保存到文件”“见下载链接”“导出完成”等说明。"
                    "如果任务尚未完成，请直接说明还缺什么，但仍然保持报告体裁。"
                ),
            },
            {"role": "user", "content": self._build_reporter_context(state["response_chunks"])},
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
        state["final_answer"] = self._sanitize_report_text(self._extract_answer(report))
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

    async def agenerate(
        self,
        prompt: str,
        workspace: str,
        temperature: float = 0.5,
        max_tokens: int = 32768,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, str]:
        """??????? AsyncOpenAI????? Web worker?"""
        workspace_abs = os.path.abspath(workspace)
        os.makedirs(workspace_abs, exist_ok=True)
        self._log_event("runtime", "run_start_async")

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
            "workspace": workspace_abs,
        }

        try:
            while (not state["finished"]) and (state["round_idx"] < state["max_rounds"]):
                planning_context = (
                    f"Task:\n{state['messages'][0]['content']}\n\n"
                    f"Last execution output:\n{state['last_execution_output'] or 'none'}\n\n"
                    f"Round: {state['round_idx']} / {state['max_rounds']}"
                )
                planning_messages: List[Dict[str, str]] = [
                    {
                        "role": "system",
                        "content": "????????????????????????????????????????????????????????????",
                    },
                    {"role": "user", "content": planning_context},
                ]
                plan = (
                    await self._achat_with_retry(
                        planning_messages,
                        state["temperature"],
                        min(state["max_tokens"], 512),
                        state["top_p"],
                        state["top_k"],
                        stop=None,
                        node="planner",
                    )
                ).strip()
                state["plan_text"] = plan
                state["response_chunks"].append(f"<Plan>\n{plan}\n</Plan>")

                coder_messages: List[Dict[str, str]] = [
                    {
                        "role": "system",
                        "content": (
                            "???????????? DeepAnalyze ???????????? <Analyze>...</Analyze>?"
                            "????????? <Code>...</Code> ?????? Python?????? <Answer>...</Answer> ???????????????"
                            "<Answer> ?????????????????????? xx??????????????????????????"
                        ),
                    },
                    {"role": "system", "content": f"Current plan:\n{state['plan_text']}"},
                ]
                if state["last_execution_output"]:
                    coder_messages.append({"role": "system", "content": f"Last execution output:\n{state['last_execution_output']}"})
                coder_messages.extend(self._compress_messages_for_coder(state))

                ans = await self._achat_with_retry(
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
                    state["final_answer"] = self._sanitize_report_text(answer_text)
                    state["finished"] = True
                    state["pending_code"] = None
                    break

                code_str = state.get("pending_code")
                if not code_str:
                    continue

                exe_output = await asyncio.to_thread(self.execute_code, code_str, state["workspace"])
                state["last_execution_output"] = exe_output
                state["response_chunks"].append(f"<Execute>\n{exe_output}\n</Execute>")
                state["messages"].append({"role": "execute", "content": exe_output})
                state["pending_code"] = None

                if exe_output.startswith("[Error]:") or exe_output.startswith("[Timeout]:"):
                    state["consecutive_exec_failures"] += 1
                    if state["consecutive_exec_failures"] > state["max_exec_retries"]:
                        break
                else:
                    state["consecutive_exec_failures"] = 0

            if not state["final_answer"]:
                reporter_messages: List[Dict[str, str]] = [
                    {
                        "role": "system",
                        "content": (
                            "????????????????????????????????????????"
                            "?????? <Answer>...</Answer> ???????"
                        ),
                    },
                    {"role": "user", "content": self._build_reporter_context(state["response_chunks"])},
                ]
                report = (
                    await self._achat_with_retry(
                        reporter_messages,
                        state["temperature"],
                        min(state["max_tokens"], 1024),
                        state["top_p"],
                        state["top_k"],
                        stop=None,
                        node="reporter",
                    )
                ).strip()
                if "<Answer>" not in report:
                    report = f"<Answer>\n{report}\n</Answer>"
                state["response_chunks"].append(report)
                state["final_answer"] = self._sanitize_report_text(self._extract_answer(report))

            reasoning = "\n".join(state["response_chunks"])
            report_body = state.get("final_answer", "").strip()
            if not report_body:
                report_body = self._sanitize_report_text(self._extract_answer(reasoning).strip() or reasoning)
            report_md = f"# ????\n\n{report_body}\n"
            report_path = self._save_report_markdown(workspace_abs, report_md)
            self._log_event("runtime", "run_end_async", rounds=state["round_idx"], finished=state["finished"], report_path=report_path)
            return {"reasoning": reasoning, "report_path": report_path}

        except Exception as err:  # noqa: BLE001
            self._log_event("runtime", "run_crashed_async", error=str(err))
            reasoning = "\n".join(state["response_chunks"])
            report_md = f"# ??????\n\n{reasoning}\n"
            report_path = self._save_report_markdown(workspace_abs, report_md)
            return {"reasoning": reasoning, "report_path": report_path}

    def generate(
        self,
        prompt: str,
        workspace: str,
        temperature: float = 0.5,
        max_tokens: int = 32768,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, str]:
        """??????? LangGraph ???????? reasoning ??????"""
        workspace_abs = os.path.abspath(workspace)
        os.makedirs(workspace_abs, exist_ok=True)
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
            "workspace": workspace_abs,
        }

        try:
            final_state = self.graph.invoke(state, config={"recursion_limit": max(200, self.max_rounds * 8)})
            reasoning = "\n".join(final_state["response_chunks"])
            report_body = final_state.get("final_answer", "").strip()
            if not report_body:
                report_body = self._sanitize_report_text(self._extract_answer(reasoning).strip() or reasoning)
            report_md = f"# ????\n\n{report_body}\n"
            report_path = self._save_report_markdown(workspace_abs, report_md)
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
            report_md = f"# ??????????\n\n{reasoning}\n"
            report_path = self._save_report_markdown(workspace_abs, report_md)

        return {"reasoning": reasoning, "report_path": report_path}


# 向后兼容旧名称
DeepAnalyzeVLLM = DeepAnalyzeLangGraph

