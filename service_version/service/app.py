import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from deepanalyze_langgraph import DeepAnalyzeLangGraph
from service.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FileDeleteResponse,
    ModelObject,
    ModelsListResponse,
)
from service.storage import Storage

load_dotenv()

API_TITLE = "DataAnalyze Service"
API_VERSION = "1.0.0"
WORKSPACE_ROOT = os.getenv("WORKSPACE", "./workspace")
DB_PATH = os.getenv("DB_PATH", os.path.join(WORKSPACE_ROOT, "storage.db"))
WEB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4.1-mini")

storage = Storage(workspace_root=WORKSPACE_ROOT, db_path=DB_PATH)


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                value = item.get("text", {}).get("value", "")
                text_parts.append(str(value))
        return "".join(text_parts)
    return str(content or "")


def _build_data_block(workspace: str) -> str:
    files = []
    for path in Path(workspace).iterdir():
        if path.is_file() and path.name != "messages.jsonl":
            files.append(path)
    if not files:
        return ""

    lines: List[str] = ["# Data"]
    for idx, path in enumerate(sorted(files), start=1):
        size_kb = path.stat().st_size / 1024
        payload = {"name": path.name, "size": f"{size_kb:.1f}KB"}
        lines.append(f"File {idx}:")
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


def _build_prompt(messages: List[Dict[str, Any]], workspace: str) -> str:
    user_texts: List[str] = []
    for msg in messages:
        if str(msg.get("role", "")).lower() == "user":
            user_texts.append(_normalize_content(msg.get("content")))

    instruction = user_texts[-1].strip() if user_texts else "请生成数据分析报告。"
    sections = ["# Instruction", instruction]
    data_block = _build_data_block(workspace)
    if data_block:
        sections.append("")
        sections.append(data_block)
    return "\n".join(sections).strip()


def _extract_answer_text(text: str) -> str:
    content = text or ""
    matches = list(re.finditer(r"<Answer>([\s\S]*?)</Answer>", content))
    if matches:
        return matches[-1].group(1).strip()
    return content.strip()


def _sanitize_report_text(text: str) -> str:
    lines: List[str] = []
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


app = FastAPI(title=API_TITLE, version=API_VERSION)
app.mount("/workspace", StaticFiles(directory=WORKSPACE_ROOT), name="workspace")
app.mount("/web", StaticFiles(directory=WEB_ROOT), name="web")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_ROOT, "index.html"))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "timestamp": int(time.time())}


@app.get("/v1/models", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    return ModelsListResponse(
        object="list",
        data=[ModelObject(id=DEFAULT_MODEL, created=int(time.time()), owned_by="deepanalyze")],
    )


@app.post("/v1/files")
async def create_file(file: UploadFile = File(...), purpose: str = Form("file-extract")) -> Dict[str, Any]:
    content = await file.read()
    file_obj = storage.create_file(file.filename, content, purpose)
    return file_obj.model_dump()


@app.get("/v1/files")
def list_files() -> Dict[str, Any]:
    return {"object": "list", "data": [f.model_dump() for f in storage.list_files()]}


@app.get("/v1/files/{file_id}")
def retrieve_file(file_id: str) -> Dict[str, Any]:
    obj = storage.get_file(file_id)
    if not obj:
        raise HTTPException(status_code=404, detail="File not found")
    return obj.model_dump()


@app.get("/v1/files/{file_id}/content")
def download_file(file_id: str) -> FileResponse:
    path = storage.get_file_path(file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File content not found")
    return FileResponse(path=path, filename=os.path.basename(path))


@app.delete("/v1/files/{file_id}", response_model=FileDeleteResponse)
def delete_file(file_id: str) -> FileDeleteResponse:
    ok = storage.delete_file(file_id)
    if not ok:
        raise HTTPException(status_code=404, detail="File not found")
    return FileDeleteResponse(id=file_id, object="file", deleted=True)


@app.get("/v1/threads")
def list_threads(owner_id: str | None = None, user_id: str | None = None) -> Dict[str, Any]:
    data: List[Dict[str, Any]] = []
    effective_owner_id = owner_id or user_id
    if not effective_owner_id:
        return {"object": "list", "data": []}
    for row in storage.list_threads(owner_id=effective_owner_id):
        data.append(
            {
                "id": row.get("id"),
                "object": "thread",
                "created_at": row.get("created_at"),
                "last_message_at": row.get("last_message_at"),
                "message_count": row.get("message_count", 0),
                "report_count": row.get("report_count", 0),
                "summary": row.get("summary", ""),
            }
        )
    return {"object": "list", "data": data}


@app.get("/v1/threads/{thread_id}")
def retrieve_thread(thread_id: str, owner_id: str | None = None, user_id: str | None = None) -> Dict[str, Any]:
    row = storage.get_thread(thread_id)
    effective_owner_id = owner_id or user_id
    if not row or not effective_owner_id or row.get("owner_id", "") != effective_owner_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"object": "thread", **row}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if req.stream:
        raise HTTPException(status_code=400, detail="当前服务版仅支持非流式请求：stream=false")

    thread_id = req.thread_id
    user_contents: List[str] = []
    for msg in req.messages:
        if str(msg.get("role", "")).lower() == "user":
            user_contents.append(_normalize_content(msg.get("content")))

    initial_summary = user_contents[0].strip() if user_contents else ""
    if thread_id and not storage.has_thread(thread_id):
        raise HTTPException(status_code=400, detail=f"Thread not found: {thread_id}")
    effective_owner_id = req.owner_id or req.user_id
    if not effective_owner_id:
        raise HTTPException(status_code=400, detail="缺少 owner_id（或 user_id）参数")
    if thread_id and req.user_id is not None:
        thread_row = storage.get_thread(thread_id)
        if thread_row and thread_row.get("owner_id", "") not in ("", effective_owner_id):
            raise HTTPException(status_code=400, detail=f"Thread not found: {thread_id}")
    if not thread_id:
        thread_id = storage.create_thread(summary=initial_summary, owner_id=effective_owner_id)
    elif initial_summary:
        storage.set_thread_summary(thread_id, initial_summary)

    workspace = storage.get_thread_workspace(thread_id)
    file_ids = req.file_ids or []
    if file_ids:
        storage.attach_files_to_thread(thread_id, file_ids)

    prompt = _build_prompt(req.messages, workspace)

    agent = DeepAnalyzeLangGraph(
        model_name=req.model or DEFAULT_MODEL,
        api_base=req.api_base or os.getenv("API_BASE"),
        api_key=req.api_key or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
        max_rounds=int(os.getenv("MAX_ROUNDS", "20")),
        max_api_retries=int(os.getenv("MAX_API_RETRIES", "3")),
        max_exec_retries=int(os.getenv("MAX_EXEC_RETRIES", "2")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    result = agent.generate(
        prompt=prompt,
        workspace=workspace,
        temperature=req.temperature or 0.5,
        max_tokens=req.max_tokens or 32768,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    reasoning = result.get("reasoning", "")
    answer_text = _sanitize_report_text(_extract_answer_text(reasoning))
    report_path = result.get("report_path", "")

    for text in user_contents:
        if text.strip():
            storage.append_thread_message(thread_id, "user", text.strip())
    if reasoning.strip():
        storage.append_thread_message(thread_id, "assistant", reasoning.strip())

    generated_files: List[Dict[str, str]] = []
    if report_path and os.path.exists(report_path):
        report_name = os.path.basename(report_path)
        report_url = f"/v1/threads/{thread_id}/reports/{report_name}"
        if effective_owner_id:
            report_url += f"?owner_id={effective_owner_id}"
        generated_files.append({"name": report_name, "url": report_url})

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": answer_text,
        "thread_id": thread_id,
    }
    if generated_files:
        message["files"] = generated_files

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model or DEFAULT_MODEL,
        choices=[ChatCompletionChoice(index=0, message=message, finish_reason="stop")],
        thread_id=thread_id,
        generated_files=generated_files if generated_files else None,
    )


@app.get("/v1/threads/{thread_id}/reports")
def list_thread_reports(thread_id: str, owner_id: str | None = None, user_id: str | None = None) -> Dict[str, Any]:
    row = storage.get_thread(thread_id)
    effective_owner_id = owner_id or user_id
    if not row or not effective_owner_id or row.get("owner_id", "") != effective_owner_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    workspace = storage.get_thread_workspace(thread_id)
    reports_dir = os.path.join(workspace, "reports")
    if not os.path.exists(reports_dir):
        return {"object": "list", "data": []}

    data: List[Dict[str, Any]] = []
    for path in sorted(Path(reports_dir).glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        report_name = path.name
        data.append(
            {
                "name": report_name,
                "url": f"/v1/threads/{thread_id}/reports/{report_name}" + (f"?owner_id={effective_owner_id}" if effective_owner_id else ""),
                "bytes": path.stat().st_size,
                "modified_at": int(path.stat().st_mtime),
            }
        )
    return {"object": "list", "data": data}


@app.get("/v1/threads/{thread_id}/reports/{report_name}")
def download_thread_report(thread_id: str, report_name: str, owner_id: str | None = None, user_id: str | None = None) -> FileResponse:
    row = storage.get_thread(thread_id)
    effective_owner_id = owner_id or user_id
    if not row or not effective_owner_id or row.get("owner_id", "") != effective_owner_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    safe_name = os.path.basename(report_name)
    workspace = storage.get_thread_workspace(thread_id)
    path = os.path.join(workspace, "reports", safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=path, filename=safe_name)


@app.get("/v1/threads/{thread_id}/messages")
def list_thread_messages(thread_id: str, limit: int = 100, owner_id: str | None = None, user_id: str | None = None) -> Dict[str, Any]:
    row = storage.get_thread(thread_id)
    effective_owner_id = owner_id or user_id
    if not row or not effective_owner_id or row.get("owner_id", "") != effective_owner_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    rows = storage.list_thread_messages(thread_id=thread_id, limit=limit)
    return {"object": "list", "data": rows}


