from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileDeleteResponse(BaseModel):
    id: str
    object: Literal["file"] = "file"
    deleted: bool


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = None


class ModelsListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]


class ChatCompletionRequest(BaseModel):
    owner_id: Optional[str] = None
    user_id: Optional[str] = None  # 兼容旧字段，等价于 owner_id
    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    messages: List[Dict[str, Any]]
    file_ids: Optional[List[str]] = None
    thread_id: Optional[str] = None
    temperature: Optional[float] = Field(default=0.5)
    max_tokens: Optional[int] = Field(default=32768)
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    thread_id: str
    generated_files: Optional[List[Dict[str, str]]] = None

