import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from service.models import FileObject


class Storage:
    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.abspath(workspace_root)
        self.files_dir = os.path.join(self.workspace_root, "_files")
        os.makedirs(self.files_dir, exist_ok=True)

        self.files: Dict[str, Dict[str, Any]] = {}
        self.threads: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_thread(self, summary: Optional[str] = None, owner_id: Optional[str] = None) -> str:
        with self._lock:
            thread_id = f"thread-{uuid.uuid4().hex[:24]}"
            workspace = self.get_thread_workspace(thread_id)
            now = int(time.time())
            self.threads[thread_id] = {
                "id": thread_id,
                "workspace": workspace,
                "created_at": now,
                "last_message_at": now,
                "message_count": 0,
                "summary": summary or "",
                "first_user_message": "",
                "owner_id": owner_id or "",
            }
            return thread_id

    def get_thread_workspace(self, thread_id: str) -> str:
        workspace = os.path.join(self.workspace_root, thread_id)
        os.makedirs(workspace, exist_ok=True)
        os.makedirs(os.path.join(workspace, "reports"), exist_ok=True)
        return workspace

    def _thread_messages_path(self, thread_id: str) -> str:
        workspace = self.get_thread_workspace(thread_id)
        return os.path.join(workspace, "messages.jsonl")

    def has_thread(self, thread_id: str) -> bool:
        with self._lock:
            return thread_id in self.threads

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self.threads.get(thread_id)
            if not item:
                return None
            return dict(item)

    @staticmethod
    def _build_summary(text: str) -> str:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            return "数据会话"
        for sep in ["。", "！", "？", ".", "!", "?", ";", "；", "\n"]:
            if sep in cleaned:
                cleaned = cleaned.split(sep, 1)[0]
                break
        cleaned = cleaned.strip()
        if len(cleaned) > 6:
            cleaned = cleaned[:6].rstrip() + "..."
        return cleaned or "数据会话"

    def list_threads(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [dict(v) for v in self.threads.values()]
        if owner_id is not None:
            rows = [row for row in rows if row.get("owner_id", "") == owner_id]
        rows.sort(key=lambda x: x.get("last_message_at", x.get("created_at", 0)), reverse=True)
        for row in rows:
            workspace = row.get("workspace", "")
            reports_dir = os.path.join(workspace, "reports") if workspace else ""
            report_count = 0
            if reports_dir and os.path.exists(reports_dir):
                report_count = len(list(Path(reports_dir).glob("*.md")))
            row["report_count"] = report_count
        return rows

    def set_thread_summary(self, thread_id: str, summary: str) -> None:
        with self._lock:
            if thread_id not in self.threads:
                return
            if not self.threads[thread_id].get("summary"):
                self.threads[thread_id]["summary"] = self._build_summary(summary)

    def attach_files_to_thread(self, thread_id: str, file_ids: List[str]) -> List[str]:
        workspace = self.get_thread_workspace(thread_id)
        copied: List[str] = []
        for file_id in file_ids:
            src = self.get_file_path(file_id)
            file_obj = self.get_file(file_id)
            if not src or not file_obj or not os.path.exists(src):
                continue
            dst = os.path.join(workspace, file_obj.filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied.append(dst)
        return copied

    def create_file(self, filename: str, content: bytes, purpose: str) -> FileObject:
        with self._lock:
            file_id = f"file-{uuid.uuid4().hex[:24]}"
            safe_name = filename.replace("\\", "_").replace("/", "_")
            ext = Path(safe_name).suffix
            disk_name = f"{file_id}{ext}"
            file_path = os.path.join(self.files_dir, disk_name)
            with open(file_path, "wb") as f:
                f.write(content)
            item = {
                "id": file_id,
                "object": "file",
                "bytes": len(content),
                "created_at": int(time.time()),
                "filename": safe_name,
                "purpose": purpose,
                "filepath": file_path,
            }
            self.files[file_id] = item
            return FileObject(**item)

    def list_files(self) -> List[FileObject]:
        with self._lock:
            return [FileObject(**v) for v in self.files.values()]

    def get_file(self, file_id: str) -> Optional[FileObject]:
        with self._lock:
            item = self.files.get(file_id)
            if not item:
                return None
            return FileObject(**item)

    def get_file_path(self, file_id: str) -> Optional[str]:
        with self._lock:
            item = self.files.get(file_id)
            return item.get("filepath") if item else None

    def delete_file(self, file_id: str) -> bool:
        with self._lock:
            item = self.files.get(file_id)
            if not item:
                return False
            path = item.get("filepath")
            if path and os.path.exists(path):
                os.remove(path)
            del self.files[file_id]
            return True

    def append_thread_message(self, thread_id: str, role: str, content: str) -> None:
        with self._lock:
            if thread_id not in self.threads:
                return
            now = int(time.time())
            item = {
                "id": f"msg-{uuid.uuid4().hex[:24]}",
                "thread_id": thread_id,
                "role": role,
                "content": content,
                "created_at": now,
            }
            path = self._thread_messages_path(thread_id)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            thread = self.threads[thread_id]
            thread["last_message_at"] = now
            thread["message_count"] = int(thread.get("message_count", 0)) + 1
            if role == "user" and not thread.get("first_user_message"):
                thread["first_user_message"] = content
                if not thread.get("summary"):
                    thread["summary"] = self._build_summary(content)

    def list_thread_messages(self, thread_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            if thread_id not in self.threads:
                return []
            path = self._thread_messages_path(thread_id)
            if not os.path.exists(path):
                return []
            rows: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if limit > 0:
                rows = rows[-limit:]
            return rows
