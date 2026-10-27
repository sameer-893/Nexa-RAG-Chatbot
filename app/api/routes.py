import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Path, Query, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.services.errors import ConflictError, NotFoundError
from app.services.rag_service import rag_service
from app.services.redis_service import redis_service


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api",
    tags=["RAG Chatbot"],
)


# ============================================================
# SHARED TYPES
# ============================================================

CHAT_ID_PATTERN = r"^[0-9a-f]{32}$"

UserId = Annotated[str, Path(min_length=1, max_length=100)]
ChatId = Annotated[str, Path(pattern=CHAT_ID_PATTERN)]
MessageId = Annotated[str, Path(min_length=1, max_length=64)]


def _http_error(exc: Exception, label: str) -> HTTPException:
    """Map service errors to HTTP errors (404 / 409 / 400 / 500)."""

    if isinstance(exc, HTTPException):
        return exc

    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=404, detail=str(exc))

    if isinstance(exc, ConflictError):
        return HTTPException(status_code=409, detail=str(exc))

    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))

    logger.exception("%s", label)

    return HTTPException(status_code=500, detail=f"{label}: {exc}")


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    user_id: str = Field(
        min_length=1,
        max_length=100,
    )

    # Omit to start a new chat; the response returns its chat_id.
    chat_id: str | None = Field(
        default=None,
        pattern=CHAT_ID_PATTERN,
    )

    question: str = Field(
        min_length=1,
        max_length=8000,
    )


class CreateChatRequest(BaseModel):

    user_id: str = Field(
        min_length=1,
        max_length=100,
    )

    title: str | None = Field(
        default=None,
        max_length=200,
    )


class ResponseUpdate(BaseModel):

    content: str = Field(
        min_length=1,
        max_length=20000,
    )


class PromptUpdate(BaseModel):

    prompt: str = Field(
        min_length=1,
        max_length=20000,
    )


class ContextLimitUpdate(BaseModel):

    context_messages: int = Field(
        ge=0,
        le=100,
    )


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health_check():

    try:
        redis_ok = redis_service.ping()

        return {
            "status": "ok",
            "redis": redis_ok,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )


# ============================================================
# DOCUMENTS
# ============================================================

# Plain `def` (not `async def`): parsing, embedding and indexing are
# blocking calls and must run in FastAPI's thread pool, not on the
# event loop.
@router.post("/documents/upload")
def upload_document(
    user_id: Annotated[
        str,
        Form(
            min_length=1,
            max_length=100,
        ),
    ],
    file: Annotated[
        UploadFile,
        File(),
    ],
):

    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:

        # Read one byte more than allowed to detect oversized files
        # without loading them completely.
        content = file.file.read(max_bytes + 1)

        if len(content) > max_bytes:

            raise HTTPException(
                status_code=413,
                detail=(
                    f"File is larger than "
                    f"{settings.max_upload_mb} MB."
                ),
            )

        return rag_service.add_file(
            user_id=user_id,
            filename=file.filename or "unknown",
            content=content,
        )

    except Exception as exc:

        raise _http_error(exc, "Upload failed")


@router.get("/documents/{user_id}")
def list_documents(
    user_id: str,
):

    try:

        return {
            "user_id": user_id,
            "documents": rag_service.get_documents(
                user_id
            ),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.get("/documents/{user_id}/{file_id}")
def document_detail(
    user_id: str,
    file_id: str,
):

    try:

        return rag_service.get_document(
            user_id,
            file_id,
        )

    except Exception as exc:

        raise _http_error(exc, "Could not load document")


@router.delete("/documents/{user_id}/{file_id}")
def delete_document(
    user_id: str,
    file_id: str,
):

    try:

        return rag_service.remove_file(
            user_id,
            file_id,
        )

    except Exception as exc:

        raise _http_error(exc, "Could not remove document")


# ============================================================
# CHATS
# ============================================================

@router.post("/chats", status_code=201)
def create_chat(
    request: CreateChatRequest,
):

    try:

        return rag_service.create_chat(
            request.user_id,
            request.title,
        )

    except Exception as exc:

        raise _http_error(exc, "Could not create chat")


@router.get("/chats/{user_id}")
def list_chats(
    user_id: UserId,
):

    try:

        return {
            "user_id": user_id,
            "chats": rag_service.list_chats(
                user_id
            ),
        }

    except Exception as exc:

        raise _http_error(exc, "Could not load chats")


# Declared before "/chats/{user_id}/{chat_id}" so "search" is never
# interpreted as a chat id.
@router.get("/chats/{user_id}/search")
def search_chats(
    user_id: UserId,
    q: Annotated[
        str,
        Query(
            min_length=1,
            max_length=200,
        ),
    ],
):

    try:

        return {
            "user_id": user_id,
            "query": q,
            "chats": rag_service.search_chats(
                user_id,
                q,
            ),
        }

    except Exception as exc:

        raise _http_error(exc, "Search failed")


@router.get("/chats/{user_id}/{chat_id}")
def get_chat(
    user_id: UserId,
    chat_id: ChatId,
):

    try:

        return rag_service.get_chat(
            user_id,
            chat_id,
        )

    except Exception as exc:

        raise _http_error(exc, "Could not load chat")


@router.delete("/chats/{user_id}/{chat_id}/messages")
def clear_chat(
    user_id: UserId,
    chat_id: ChatId,
):
    """Clear the messages of ONE chat. The chat stays in the list."""

    try:

        rag_service.clear_chat(
            user_id,
            chat_id,
        )

        return {
            "message": "Chat cleared.",
            "chat_id": chat_id,
        }

    except Exception as exc:

        raise _http_error(exc, "Could not clear chat")


@router.delete("/chats/{user_id}/{chat_id}")
def delete_chat(
    user_id: UserId,
    chat_id: ChatId,
):
    """Delete ONE chat completely."""

    try:

        rag_service.delete_chat(
            user_id,
            chat_id,
        )

        return {
            "message": "Chat deleted.",
            "chat_id": chat_id,
        }

    except Exception as exc:

        raise _http_error(exc, "Could not delete chat")


@router.delete("/chats/{user_id}")
def delete_all_chats(
    user_id: UserId,
):
    """Delete ALL chats of a user. Uploaded documents are not touched."""

    try:

        deleted = rag_service.delete_all_chats(
            user_id
        )

        return {
            "message": "All chats deleted.",
            "user_id": user_id,
            "deleted_chats": deleted,
        }

    except Exception as exc:

        raise _http_error(exc, "Could not delete chats")


# ============================================================
# RESPONSES (edit / restore)
# ============================================================

@router.put("/chats/{user_id}/{chat_id}/messages/{message_id}")
def update_response(
    user_id: UserId,
    chat_id: ChatId,
    message_id: MessageId,
    request: ResponseUpdate,
):

    try:

        return rag_service.update_response(
            user_id,
            chat_id,
            message_id,
            request.content,
        )

    except Exception as exc:

        raise _http_error(exc, "Could not update response")


@router.post("/chats/{user_id}/{chat_id}/messages/{message_id}/restore")
def restore_response(
    user_id: UserId,
    chat_id: ChatId,
    message_id: MessageId,
):

    try:

        return rag_service.restore_response(
            user_id,
            chat_id,
            message_id,
        )

    except Exception as exc:

        raise _http_error(exc, "Could not restore response")


# ============================================================
# CHAT
# ============================================================

@router.post("/chat/query")
def chat_query(
    request: ChatRequest,
):

    try:

        return rag_service.chat(
            user_id=request.user_id,
            question=request.question,
            chat_id=request.chat_id,
        )

    except Exception as exc:

        raise _http_error(exc, "Chat failed")


# ============================================================
# PROMPT
# ============================================================

@router.get("/config/prompt")
def get_prompt():

    return {
        "prompt": rag_service.get_prompt()
    }


@router.put("/config/prompt")
def update_prompt(
    request: PromptUpdate,
):

    try:

        prompt = rag_service.update_prompt(
            request.prompt
        )

        return {
            "prompt": prompt
        }

    except Exception as exc:

        raise _http_error(exc, "Could not save prompt")


@router.delete("/config/prompt")
def reset_prompt():
    """Restore the built-in default prompt."""

    return {
        "prompt": rag_service.reset_prompt()
    }


# ============================================================
# CONTEXT
# ============================================================

@router.get("/config/context")
def get_context_limit():

    return {
        "context_messages": (
            rag_service.get_context_limit()
        )
    }


@router.put("/config/context")
def update_context_limit(
    request: ContextLimitUpdate,
):

    try:

        value = (
            rag_service.update_context_limit(
                request.context_messages
            )
        )

        return {
            "context_messages": value
        }

    except Exception as exc:

        raise _http_error(exc, "Could not save context limit")
