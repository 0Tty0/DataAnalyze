import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from service.models import FileObject


class Storage:
    def __init__(self, workspace_root: str, db_path: Optional[str] = None):
        self.workspace_root = os.path.abspath(workspace_root)
        self.files_dir = os.path.join(self.workspace_root, "_files")
        os.makedirs(self.files_dir, exist_ok=True)

        self.db_path = db_path or os.path.join(self.workspace_root, "storage.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    workspace TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_message_at INTEGER NOT NULL,
                    message_count INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    first_user_message TEXT NOT NULL,
                    owner_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    object TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    filepath TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_threads_owner_last_msg ON threads(owner_id, last_message_at DESC)"
            )
            conn.commit()

    def create_thread(self, summary: Optional[str] = None, owner_id: Optional[str] = None) -> str:
        with self._lock:
            thread_id = f"thread-{uuid.uuid4().hex[:24]}"
            workspace = self.get_thread_workspace(thread_id)
            now = int(time.time())
            row = {
                "id": thread_id,
                "workspace": workspace,
                "created_at": now,
                "last_message_at": now,
                "message_count": 0,
                "summary": self._build_summary(summary or "") if summary else "",
                "first_user_message": "",
                "owner_id": owner_id or "",
            }
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO threads(id, workspace, created_at, last_message_at, message_count, summary, first_user_message, owner_id)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        row["workspace"],
                        row["created_at"],
                        row["last_message_at"],
                        row["message_count"],
                        row["summary"],
                        row["first_user_message"],
                        row["owner_id"],
                    ),
                )
                conn.commit()
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
            with self._connect() as conn:
                row = conn.execute("SELECT 1 FROM threads WHERE id = ? LIMIT 1", (thread_id,)).fetchone()
                return row is not None

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            return None
        return dict(row)

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
            with self._connect() as conn:
                if owner_id is None:
                    rows = conn.execute(
                        "SELECT * FROM threads ORDER BY last_message_at DESC, created_at DESC"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM threads WHERE owner_id = ? ORDER BY last_message_at DESC, created_at DESC",
                        (owner_id,),
                    ).fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            workspace = item.get("workspace", "")
            reports_dir = os.path.join(workspace, "reports") if workspace else ""
            report_count = 0
            if reports_dir and os.path.exists(reports_dir):
                report_count = len(list(Path(reports_dir).glob("*.md")))
            item["report_count"] = report_count
            result.append(item)
        return result

    def set_thread_summary(self, thread_id: str, summary: str) -> None:
        built = self._build_summary(summary)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE threads
                    SET summary = CASE WHEN summary = '' THEN ? ELSE summary END
                    WHERE id = ?
                    """,
                    (built, thread_id),
                )
                conn.commit()

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

            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO files(id, object, bytes, created_at, filename, purpose, filepath)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["id"],
                        item["object"],
                        item["bytes"],
                        item["created_at"],
                        item["filename"],
                        item["purpose"],
                        item["filepath"],
                    ),
                )
                conn.commit()

            return FileObject(**item)

    def list_files(self) -> List[FileObject]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM files ORDER BY created_at DESC").fetchall()
        return [FileObject(**dict(row)) for row in rows]

    def get_file(self, file_id: str) -> Optional[FileObject]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        if not row:
            return None
        return FileObject(**dict(row))

    def get_file_path(self, file_id: str) -> Optional[str]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT filepath FROM files WHERE id = ?", (file_id,)).fetchone()
        if not row:
            return None
        return str(row["filepath"])

    def delete_file(self, file_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT filepath FROM files WHERE id = ?", (file_id,)).fetchone()
                if not row:
                    return False
                path = str(row["filepath"])
                conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                conn.commit()

        if path and os.path.exists(path):
            os.remove(path)
        return True

    def append_thread_message(self, thread_id: str, role: str, content: str) -> None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT id, message_count, first_user_message, summary FROM threads WHERE id = ?",
                    (thread_id,),
                ).fetchone()
                if not row:
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

                first_user_message = str(row["first_user_message"] or "")
                summary = str(row["summary"] or "")
                if role == "user" and not first_user_message:
                    first_user_message = content
                    if not summary:
                        summary = self._build_summary(content)

                new_message_count = int(row["message_count"] or 0) + 1
                conn.execute(
                    """
                    UPDATE threads
                    SET last_message_at = ?, message_count = ?, first_user_message = ?, summary = ?
                    WHERE id = ?
                    """,
                    (now, new_message_count, first_user_message, summary, thread_id),
                )
                conn.commit()

    def list_thread_messages(self, thread_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT 1 FROM threads WHERE id = ? LIMIT 1", (thread_id,)).fetchone()
                if not row:
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
