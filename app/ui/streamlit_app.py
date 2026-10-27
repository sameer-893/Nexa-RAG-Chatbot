"""
Nexa RAG - chat interface.

Main area : Chat | Responses | Settings (segmented tabs) and, on the Chat tab,
            the message box with a file-upload button.
Sidebar   : new chat, search, every saved chat, uploaded documents.

The UI is a thin client: every action goes through the FastAPI backend
(app/api/routes.py). Requires Streamlit >= 1.43 (st.chat_input accept_file).
"""

from __future__ import annotations

import html
import os
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


API_BASE_URL = os.getenv("NEXA_API_URL", "http://127.0.0.1:8000").rstrip("/")

# The backend has no authentication; the user id is the only data boundary.
DEFAULT_USER_ID = os.getenv("NEXA_USER_ID", "user1").strip() or "user1"
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]{1,100}$")

ALLOWED_FILE_TYPES = ["pdf", "docx", "txt", "md"]
MAX_UPLOAD_BYTES = _env_int("NEXA_MAX_UPLOAD_MB", 20) * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120

MAX_PROMPT_CHARS = 20000
RESPONSES_PAGE_SIZE = 10

ROUTE_LABELS = {
    "direct": "Direct LLM",
    "rag": "Knowledge base",
    "web": "Web search",
}

VIEW_CHAT = "💬 Chat"
VIEW_RESPONSES = "✎ Responses"
VIEW_SETTINGS = "⚙ Settings"
VIEWS = [VIEW_CHAT, VIEW_RESPONSES, VIEW_SETTINGS]


st.set_page_config(
    page_title="Nexa RAG",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Theme-neutral CSS: no hard-coded backgrounds, so light/dark themes both work.
st.markdown(
    """
    <style>
        #MainMenu, footer { visibility: hidden; }
        .chip {
            display: inline-block;
            margin: 6px 6px 0 0;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            background: rgba(128, 128, 128, 0.16);
        }
        .chip a { color: inherit; text-decoration: none; }
        .chip-route { background: rgba(98, 84, 230, 0.20); }
        .chip-edited { background: rgba(230, 160, 40, 0.25); }
        /* chat list: left-aligned titles, highlighted active chat */
        [class*="st-key-chat_"] button,
        [class*="st-key-chatactive_"] button {
            justify-content: flex-start;
            text-align: left;
        }
        [class*="st-key-chatactive_"] button {
            border-color: rgba(98, 84, 230, 0.65);
            background: rgba(98, 84, 230, 0.14);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE: dict[str, Any] = {
    "user_id": DEFAULT_USER_ID,
    "view": VIEW_CHAT,
    "pending_view": None,        # applied before the tab widget is created
    # chats
    "chats": [],
    "chats_loaded": False,
    "chats_error": "",
    "chats_autoopened": False,
    "active_chat_id": None,
    "messages": [],              # messages of the open chat (+ ephemeral errors)
    "chat_loaded_id": None,      # which chat `messages` belongs to
    "chat_error": "",
    "search_cache": None,
    # documents
    "documents": [],
    "documents_loaded": False,
    "documents_error": "",
    "doc_detail": None,
    "doc_detail_error": "",
    # settings
    "prompt": "",
    "context_messages": None,
    "settings_loaded": False,
    "settings_error": "",
    "settings_version": 0,       # bumping it re-creates the settings inputs
    # responses tab
    "responses_limit": RESPONSES_PAGE_SIZE,
    "responses_limit_chat": None,
    # one-shot messages shown at the top of the main area
    "flash": [],
}


def _fresh(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


for _key, _value in DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _fresh(_value)


def reset_workspace_state() -> None:
    """Forget everything loaded for the previous workspace/user."""

    keep = {"user_id", "view", "pending_view"}

    for key, value in DEFAULT_STATE.items():
        if key not in keep:
            st.session_state[key] = _fresh(value)


def current_user() -> str:
    return st.session_state.user_id


def push_flash(kind: str, text: str) -> None:
    st.session_state.flash.append((kind, text))


# ============================================================
# BACKEND CLIENT
# ============================================================

class ApiError(RuntimeError):
    """A backend call failed; the message is safe to show to the user."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def api_request(method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
    """Call the FastAPI backend and return the decoded JSON object."""

    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.request(
            method,
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise ApiError(
            f"Cannot reach the backend at {API_BASE_URL} "
            f"({exc.__class__.__name__})."
        ) from exc

    try:
        data = response.json()
    except ValueError:
        data = {}

    if not isinstance(data, dict):
        data = {"data": data}

    if not response.ok:
        detail = data.get("detail") or data.get("message")

        if isinstance(detail, list):
            # FastAPI validation errors arrive as a list of dicts.
            detail = "; ".join(
                str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                for item in detail
            )

        raise ApiError(
            str(detail or f"Request failed ({response.status_code})."),
            status=response.status_code,
        )

    return data


def _path(prefix: str, *parts: str) -> str:
    segments = [quote(current_user(), safe="")]
    segments += [quote(part, safe="") for part in parts]
    return f"{prefix}/" + "/".join(segments)


# ============================================================
# DATA ACTIONS: CHATS
# ============================================================

def load_chats() -> None:
    try:
        data = api_request("GET", _path("/api/chats"))
    except ApiError as exc:
        st.session_state.chats_error = str(exc)
        return  # chats_loaded stays False, so the next rerun retries

    chats = data.get("chats")
    st.session_state.chats = chats if isinstance(chats, list) else []
    st.session_state.chats_loaded = True
    st.session_state.chats_error = ""
    st.session_state.search_cache = None


def load_chat_messages() -> None:
    """Restore the messages of the active chat from the backend."""

    chat_id = st.session_state.active_chat_id

    if not chat_id:
        st.session_state.messages = []
        st.session_state.chat_loaded_id = None
        return

    try:
        data = api_request("GET", _path("/api/chats", chat_id))
    except ApiError as exc:
        if exc.status == 404:  # deleted elsewhere
            st.session_state.active_chat_id = None
            st.session_state.messages = []
            st.session_state.chat_loaded_id = None
            st.session_state.chats_loaded = False
            push_flash("warning", "That chat no longer exists.")
            return

        st.session_state.chat_error = str(exc)
        st.session_state.messages = []
        st.session_state.chat_loaded_id = chat_id  # do not retry on every rerun
        return

    messages = data.get("messages")
    st.session_state.messages = messages if isinstance(messages, list) else []
    st.session_state.chat_loaded_id = chat_id
    st.session_state.chat_error = ""


def open_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id
    st.session_state.view = VIEW_CHAT
    st.session_state.chat_loaded_id = None  # forces a reload on this run


def create_chat() -> str | None:
    try:
        chat = api_request("POST", "/api/chats", json={"user_id": current_user()})
    except ApiError as exc:
        push_flash("error", f"Could not create a chat: {exc}")
        return None

    st.session_state.chats.insert(0, chat)
    st.session_state.search_cache = None
    return chat["chat_id"]


def start_new_chat() -> None:
    """Open a fresh chat; an already empty chat is reused, not duplicated."""

    active = find_chat(st.session_state.active_chat_id)

    if active is not None and int(active.get("message_count", 0)) == 0:
        open_chat(active["chat_id"])
        return

    chat_id = create_chat()

    if chat_id:
        open_chat(chat_id)


def find_chat(chat_id: str | None) -> dict[str, Any] | None:
    for chat in st.session_state.chats:
        if chat.get("chat_id") == chat_id:
            return chat

    return None


def clear_active_chat() -> None:
    chat_id = st.session_state.active_chat_id

    try:
        api_request("DELETE", _path("/api/chats", chat_id, "messages"))
    except ApiError as exc:
        push_flash("error", f"Could not clear the chat: {exc}")
        return

    st.session_state.messages = []
    st.session_state.chat_loaded_id = chat_id
    st.session_state.chats_loaded = False
    push_flash("success", "Chat cleared. Other chats and documents were not changed.")


def delete_active_chat() -> None:
    chat_id = st.session_state.active_chat_id

    try:
        api_request("DELETE", _path("/api/chats", chat_id))
    except ApiError as exc:
        push_flash("error", f"Could not delete the chat: {exc}")
        return

    st.session_state.active_chat_id = None
    st.session_state.messages = []
    st.session_state.chat_loaded_id = None
    st.session_state.chats_loaded = False
    push_flash("success", "Chat deleted.")


def delete_all_chats() -> None:
    try:
        result = api_request("DELETE", _path("/api/chats"))
    except ApiError as exc:
        push_flash("error", f"Could not delete chats: {exc}")
        return

    st.session_state.active_chat_id = None
    st.session_state.messages = []
    st.session_state.chat_loaded_id = None
    st.session_state.chats = []
    st.session_state.chats_loaded = False
    push_flash(
        "success",
        f"Deleted {result.get('deleted_chats', 0)} chat(s). "
        "Documents and settings were not changed.",
    )


def visible_chats(query: str) -> tuple[list[dict[str, Any]], str]:
    """Chats to list in the sidebar, honouring the search box."""

    needle = query.strip()

    if not needle:
        return st.session_state.chats, ""

    cache = st.session_state.search_cache

    if cache and cache["query"] == needle:
        return cache["results"], cache["note"]

    note = ""

    try:
        data = api_request(
            "GET",
            _path("/api/chats", "search"),
            params={"q": needle},
        )
        results = data.get("chats", [])
    except ApiError:
        # Fall back to title matching so search keeps working offline.
        results = [
            chat for chat in st.session_state.chats
            if needle.lower() in str(chat.get("title", "")).lower()
        ]
        note = "Search is limited to chat titles right now."

    st.session_state.search_cache = {"query": needle, "results": results, "note": note}
    return results, note


# ============================================================
# DATA ACTIONS: MESSAGES AND RESPONSES
# ============================================================

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ask(question: str) -> None:
    """Send a question to the active chat and append the saved messages."""

    chat_id = st.session_state.active_chat_id

    try:
        result = api_request(
            "POST",
            "/api/chat/query",
            json={"user_id": current_user(), "chat_id": chat_id, "question": question},
        )
    except ApiError as exc:
        st.session_state.messages.append(
            {"role": "user", "content": question, "timestamp": now_iso()}
        )
        st.session_state.messages.append({"role": "error", "content": str(exc)})
        return

    st.session_state.messages.append(result["user_message"])
    st.session_state.messages.append(result["assistant_message"])


def save_response(chat_id: str, message_id: str, content: str) -> None:
    api_request(
        "PUT",
        _path("/api/chats", chat_id, "messages", message_id),
        json={"content": content},
    )
    st.session_state.chat_loaded_id = None  # chat view reloads the edit


def restore_response(chat_id: str, message_id: str) -> None:
    api_request("POST", _path("/api/chats", chat_id, "messages", message_id, "restore"))
    st.session_state.chat_loaded_id = None


# ============================================================
# DATA ACTIONS: DOCUMENTS
# ============================================================

def load_documents() -> None:
    try:
        data = api_request("GET", _path("/api/documents"))
    except ApiError as exc:
        st.session_state.documents_error = str(exc)
        return

    documents = data.get("documents")
    st.session_state.documents = documents if isinstance(documents, list) else []
    st.session_state.documents_loaded = True
    st.session_state.documents_error = ""


def show_document_detail(file_id: str) -> None:
    st.session_state.doc_detail_error = ""

    try:
        st.session_state.doc_detail = api_request(
            "GET", _path("/api/documents", file_id)
        )
    except ApiError as exc:
        st.session_state.doc_detail = None
        st.session_state.doc_detail_error = str(exc)


def remove_document(file_id: str, filename: str) -> None:
    try:
        api_request("DELETE", _path("/api/documents", file_id))
    except ApiError as exc:
        push_flash("error", f"Could not remove {filename}: {exc}")
        return

    detail = st.session_state.doc_detail

    if detail and detail.get("file_id") == file_id:
        st.session_state.doc_detail = None

    st.session_state.documents_loaded = False
    push_flash("success", f"Removed {filename} and its indexed content.")


def upload_file(uploaded_file: Any) -> dict[str, Any]:
    content = uploaded_file.getvalue()

    if not content:
        raise ApiError("The file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ApiError(f"The file is larger than {limit_mb} MB.")

    return api_request(
        "POST",
        "/api/documents/upload",
        data={"user_id": current_user()},
        files={
            "file": (
                uploaded_file.name,
                content,
                uploaded_file.type or "application/octet-stream",
            )
        },
    )


def index_files(files: list[Any]) -> None:
    """Upload files one by one; report each result."""

    for uploaded_file in files:
        try:
            result = upload_file(uploaded_file)
        except ApiError as exc:
            push_flash("error", f"Could not index **{uploaded_file.name}**: {exc}")
            continue

        push_flash(
            "success",
            f"Indexed **{result.get('filename', uploaded_file.name)}** "
            f"({result.get('chunk_count', 0)} chunks).",
        )

    st.session_state.documents_loaded = False


# ============================================================
# DATA ACTIONS: SETTINGS
# ============================================================

def load_settings() -> None:
    try:
        prompt = api_request("GET", "/api/config/prompt")
        context = api_request("GET", "/api/config/context")
    except ApiError as exc:
        st.session_state.settings_error = str(exc)
        return

    st.session_state.prompt = str(prompt.get("prompt", ""))
    st.session_state.context_messages = int(context.get("context_messages", 0))
    st.session_state.settings_loaded = True
    st.session_state.settings_error = ""
    st.session_state.settings_version += 1


# ============================================================
# HELPERS
# ============================================================

def format_time(timestamp: Any) -> str:
    try:
        moment = datetime.fromisoformat(str(timestamp))
    except ValueError:
        return ""

    if moment.tzinfo is not None:
        moment = moment.astimezone()

    return moment.strftime("%b %d, %H:%M")


def shorten(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def persisted_messages() -> list[dict[str, Any]]:
    return [
        message
        for message in st.session_state.messages
        if isinstance(message, dict) and message.get("message_id")
    ]


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


# ============================================================
# RENDERING: MESSAGES
# ============================================================

def render_chips(message: dict[str, Any]) -> None:
    """Route badge + de-duplicated source chips (all values HTML-escaped)."""

    chips: list[str] = []

    route = message.get("route")
    if route:
        label = ROUTE_LABELS.get(str(route), str(route))
        chips.append(f'<span class="chip chip-route">{html.escape(label)}</span>')

    if message.get("edited"):
        chips.append('<span class="chip chip-edited">✎ edited</span>')

    sources = message.get("sources")
    seen: set[str] = set()

    for source in sources if isinstance(sources, list) else []:
        if not isinstance(source, dict):
            continue

        if source.get("type") == "web":
            url = str(source.get("url", ""))

            if not url.startswith(("http://", "https://")) or url in seen:
                continue

            seen.add(url)
            title = shorten(str(source.get("title") or url))
            chips.append(
                '<span class="chip">'
                f'<a href="{html.escape(url)}" target="_blank" '
                f'rel="noopener noreferrer">🌐 {html.escape(title)}</a></span>'
            )
        else:
            name = str(source.get("filename") or "Document")

            if name in seen:
                continue

            seen.add(name)
            chips.append(f'<span class="chip">📄 {html.escape(shorten(name))}</span>')

    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role")
    content = str(message.get("content", ""))
    when = format_time(message.get("timestamp"))

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
            if when:
                st.caption(when)

    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)
            render_chips(message)
            if when:
                st.caption(when)

    elif role == "error":
        st.error(content, icon="⚠️")


def render_flash() -> None:
    """Show (and clear) one-shot success/warning/error messages."""

    flashes, st.session_state.flash = st.session_state.flash, []

    for kind, text in flashes:
        if kind == "success":
            st.success(text, icon="✅")
        elif kind == "warning":
            st.warning(text, icon="⚠️")
        else:
            st.error(text, icon="⚠️")


# ============================================================
# SIDEBAR
# ============================================================

def render_chat_list(search_text: str) -> None:
    chats, note = visible_chats(search_text)

    if st.session_state.chats_error and not st.session_state.chats_loaded:
        st.caption(f"⚠️ Could not load chats: {st.session_state.chats_error}")
        st.button("Retry", key="retry_chats")
        return

    if not chats:
        if search_text.strip():
            st.caption(f"No chats match “{search_text.strip()}”.")
        else:
            st.caption("No chats yet. Start one with ＋ New chat.")
        return

    if search_text.strip():
        st.caption(f"{plural(len(chats), 'chat')} found")

    if note:
        st.caption(note)

    with st.container(height=280, border=False):
        for chat in chats:
            chat_id = chat["chat_id"]
            active = chat_id == st.session_state.active_chat_id
            help_text = f"{plural(int(chat.get('message_count', 0)), 'message')}"

            if chat.get("updated_at"):
                help_text += f" · {format_time(chat['updated_at'])}"

            if chat.get("snippet"):
                help_text += f"\n…{chat['snippet']}…"

            if st.button(
                shorten(str(chat.get("title", "New chat")), 30),
                key=f"{'chatactive' if active else 'chat'}_{chat_id}",
                help=help_text,
                use_container_width=True,
            ):
                open_chat(chat_id)
                st.rerun()


def render_documents() -> None:
    st.markdown("**Documents**")

    if st.session_state.documents_error and not st.session_state.documents_loaded:
        st.caption(f"⚠️ Could not load documents: {st.session_state.documents_error}")
        return

    documents = st.session_state.documents

    if not documents:
        st.caption("No documents yet. Use the + button in the message box.")

    for document in documents:
        if not isinstance(document, dict):
            continue

        file_id = str(document.get("file_id", ""))
        filename = str(document.get("filename", "Document"))
        chunks = int(document.get("chunk_count", 0))

        name_col, info_col, remove_col = st.columns([5, 1, 1], vertical_alignment="center")
        name_col.caption(f"📄 {shorten(filename, 24)}")

        if info_col.button("ⓘ", key=f"info_{file_id}", help=f"Details for {filename}"):
            show_document_detail(file_id)

        with remove_col.popover("✕", help=f"Remove {filename}"):
            st.markdown(f"**Remove “{shorten(filename, 40)}”?**")
            st.caption(
                f"Deletes the file and its {plural(chunks, 'indexed chunk')} from the "
                "knowledge base. Your chats are not affected."
            )

            if st.button("Remove file", key=f"remove_{file_id}", type="primary"):
                remove_document(file_id, filename)
                st.rerun()

    render_document_detail()


def render_document_detail() -> None:
    if st.session_state.doc_detail_error:
        st.caption(f"⚠️ {st.session_state.doc_detail_error}")
        return

    detail = st.session_state.doc_detail

    if not detail:
        return

    with st.container(border=True):
        st.markdown(f"**{detail.get('filename', 'Document')}**")
        st.caption(
            f"Chunks: {detail.get('chunk_count', '?')}  \n"
            f"Size: {int(detail.get('size_bytes', 0)) / 1024:.1f} KB  \n"
            f"Uploaded: {format_time(detail.get('uploaded_at')) or 'unknown'}  \n"
            f"Indexed: {'yes ✅' if detail.get('vector_exists') else 'not found ⚠️'}  \n"
            f"ID: {detail.get('file_id', '')}"
        )

        if st.button("Close", key="close_detail"):
            st.session_state.doc_detail = None
            st.rerun()


def render_sidebar() -> str:
    """Draw the sidebar and return the current search text."""

    with st.sidebar:
        st.markdown("### 💬 Nexa RAG")

        if st.button("＋ New chat", key="new_chat", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()

        search_text = st.text_input(
            "Search chats",
            placeholder="🔍 Search chats",
            label_visibility="collapsed",
            key="chat_search",
        )

        render_chat_list(search_text)

        st.divider()
        render_documents()

    return search_text


# ============================================================
# MAIN: CHAT TAB
# ============================================================

def render_chat_header(chat: dict[str, Any]) -> None:
    messages = persisted_messages()
    total = len(messages)
    title = str(chat.get("title", "New chat"))

    title_col, clear_col, delete_col = st.columns([4, 1.5, 1.5], vertical_alignment="center")
    title_col.markdown(f"#### {shorten(title, 48)}")

    with clear_col.popover("🧹 Clear", use_container_width=True, disabled=total == 0):
        st.markdown(f"**Clear “{shorten(title, 40)}”?**")
        st.caption(
            f"Removes all {plural(total, 'message')} in this chat only. The chat stays in "
            "your list. Other chats and uploaded documents are not affected."
        )

        if st.button("Yes, clear this chat", key="confirm_clear", type="primary"):
            clear_active_chat()
            st.rerun()

    with delete_col.popover("🗑 Delete", use_container_width=True):
        st.markdown(f"**Delete “{shorten(title, 40)}”?**")
        st.caption(
            f"Permanently deletes this chat and its {plural(total, 'message')}. "
            "Other chats and uploaded documents are not affected."
        )

        if st.button("Yes, delete this chat", key="confirm_delete", type="primary"):
            delete_active_chat()
            st.rerun()

    limit = st.session_state.context_messages

    if total and limit is not None:
        st.caption(
            f"{plural(total, 'message')} saved and shown in full · "
            f"the model sees only the latest {min(limit, total)} as context"
        )


def render_chat_tab() -> None:
    chat_id = st.session_state.active_chat_id

    if chat_id and st.session_state.chat_loaded_id != chat_id:
        with st.spinner("Loading chat…"):
            load_chat_messages()

    chat = find_chat(st.session_state.active_chat_id)

    render_flash()

    if st.session_state.active_chat_id and chat is not None:
        render_chat_header(chat)

    if st.session_state.chat_error:
        st.error(f"Could not load this chat: {st.session_state.chat_error}")

    messages = st.session_state.messages

    if not messages and not st.session_state.chat_error:
        st.markdown("### What can I help with?")
        st.caption(
            "Ask a question, or use the + button in the message box to add a "
            "PDF, DOCX, TXT or Markdown file to your knowledge base."
        )

    for message in messages:
        if isinstance(message, dict):
            render_message(message)

    handle_input()


def handle_input() -> None:
    """Message box with an upload button. Must stay at top level to pin to the bottom."""

    try:
        submission = st.chat_input(
            "Ask anything, or add a file with +",
            accept_file="multiple",
            file_type=ALLOWED_FILE_TYPES,
        )
    except TypeError:
        st.error(
            "This interface needs Streamlit 1.43 or newer "
            "(file upload inside the message box). "
            "Run: pip install -U streamlit"
        )
        st.stop()

    if not submission:
        return

    text = str(submission.get("text") or "").strip()
    files = list(submission.get("files") or [])

    if files:
        with st.spinner("Indexing files…"):
            index_files(files)  # first, so the question can use the new files

    if text:
        if not st.session_state.active_chat_id:
            chat_id = create_chat()

            if chat_id is None:
                st.rerun()

            st.session_state.active_chat_id = chat_id
            st.session_state.messages = []
            st.session_state.chat_loaded_id = chat_id

        with st.chat_message("user"):
            st.markdown(text)

        with st.spinner("Thinking…"):
            ask(text)

        st.session_state.chats_loaded = False  # title, order and counts changed

    st.rerun()


# ============================================================
# MAIN: RESPONSES TAB
# ============================================================

def render_responses_tab() -> None:
    st.markdown("### Responses")
    st.caption(
        "Edit what the assistant answered. A saved edit replaces the response in the "
        "chat and in the context sent to the model for later messages. "
        "The original text is kept and can be restored."
    )

    render_flash()

    chats = st.session_state.chats

    if not chats:
        st.info("No chats yet. Responses appear here after you chat.")
        return

    chat_ids = [chat["chat_id"] for chat in chats]

    if st.session_state.get("responses_chat") not in chat_ids:
        active = st.session_state.active_chat_id
        st.session_state.responses_chat = active if active in chat_ids else chat_ids[0]

    titles = {chat["chat_id"]: chat for chat in chats}

    selected = st.selectbox(
        "Chat",
        chat_ids,
        key="responses_chat",
        format_func=lambda cid: (
            f"{shorten(str(titles[cid].get('title', 'New chat')), 50)} "
            f"({plural(int(titles[cid].get('message_count', 0)), 'message')})"
        ),
    )

    if st.session_state.responses_limit_chat != selected:
        st.session_state.responses_limit_chat = selected
        st.session_state.responses_limit = RESPONSES_PAGE_SIZE

    try:
        with st.spinner("Loading responses…"):
            data = api_request("GET", _path("/api/chats", selected))
    except ApiError as exc:
        st.error(f"Could not load this chat: {exc}")
        return

    messages = [m for m in data.get("messages", []) if isinstance(m, dict)]

    items = []

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        previous = messages[index - 1] if index > 0 else {}
        question = previous.get("content", "") if previous.get("role") == "user" else ""
        items.append((question, message))

    if not items:
        st.info("This chat has no assistant responses yet.")
        return

    items.reverse()  # newest first
    shown = items[: st.session_state.responses_limit]

    for question, message in shown:
        render_response_editor(selected, question, message)

    if len(items) > len(shown):
        if st.button(f"Show older responses ({len(items) - len(shown)} more)", key="more_responses"):
            st.session_state.responses_limit += RESPONSES_PAGE_SIZE
            st.rerun()


def render_response_editor(chat_id: str, question: str, message: dict[str, Any]) -> None:
    message_id = str(message["message_id"])
    stored = str(message.get("content", ""))
    edited = bool(message.get("edited"))
    version = message.get("edited_at") or "original"

    with st.container(border=True):
        if question:
            st.caption(f"Question · {format_time(message.get('timestamp'))}")
            st.markdown(f"**{shorten(question, 200)}**")

        new_text = st.text_area(
            "Response",
            value=stored,
            key=f"resp_{message_id}_{version}",
            height=160,
            label_visibility="collapsed",
        )

        if edited:
            st.caption(f"✎ Edited {format_time(message.get('edited_at'))}")

        changed = new_text.strip() != stored.strip()

        save_col, restore_col, _ = st.columns([2, 2, 3])

        if save_col.button(
            "Save changes",
            key=f"save_{message_id}",
            type="primary",
            disabled=not changed or not new_text.strip(),
        ):
            try:
                save_response(chat_id, message_id, new_text)
            except ApiError as exc:
                st.error(f"Could not save: {exc}")
            else:
                push_flash("success", "Response saved.")
                st.rerun()

        if edited and restore_col.button("Restore original", key=f"restore_{message_id}"):
            try:
                restore_response(chat_id, message_id)
            except ApiError as exc:
                st.error(f"Could not restore: {exc}")
            else:
                push_flash("success", "Original response restored.")
                st.rerun()


# ============================================================
# MAIN: SETTINGS TAB
# ============================================================

def render_settings_tab() -> None:
    st.markdown("### Settings")
    render_flash()

    if st.session_state.settings_error and not st.session_state.settings_loaded:
        st.error(f"Could not load settings: {st.session_state.settings_error}")
        st.button("Retry", key="retry_settings")
        return

    version = st.session_state.settings_version

    # ---- prompt -------------------------------------------------
    st.markdown("**Response prompt**")
    st.caption("The system prompt used to generate every response, for all chats.")

    prompt_text = st.text_area(
        "Prompt",
        value=st.session_state.prompt,
        key=f"prompt_editor_{version}",
        height=260,
        max_chars=MAX_PROMPT_CHARS,
        label_visibility="collapsed",
    )

    save_col, reset_col, _ = st.columns([2, 2, 3])

    if save_col.button(
        "Save prompt",
        key="save_prompt",
        type="primary",
        disabled=prompt_text.strip() == st.session_state.prompt.strip()
        or not prompt_text.strip(),
    ):
        try:
            data = api_request("PUT", "/api/config/prompt", json={"prompt": prompt_text})
        except ApiError as exc:
            st.error(f"Could not save the prompt: {exc}")
        else:
            st.session_state.prompt = str(data.get("prompt", prompt_text))
            st.session_state.settings_version += 1
            push_flash("success", "Prompt saved. It applies to the next response.")
            st.rerun()

    if reset_col.button("Reset to default", key="reset_prompt"):
        try:
            data = api_request("DELETE", "/api/config/prompt")
        except ApiError as exc:
            st.error(f"Could not reset the prompt: {exc}")
        else:
            st.session_state.prompt = str(data.get("prompt", ""))
            st.session_state.settings_version += 1
            push_flash("success", "Prompt reset to the default.")
            st.rerun()

    st.divider()

    # ---- context -----------------------------------------------
    st.markdown("**Conversation context**")
    st.caption(
        "How many of the latest messages of a chat are sent to the model. "
        "This never limits what is saved or shown: chats keep their complete history."
    )

    context_value = st.number_input(
        "Messages sent as context",
        min_value=0,
        max_value=100,
        value=int(st.session_state.context_messages or 0),
        step=1,
        key=f"context_input_{version}",
    )

    if st.button(
        "Save context size",
        key="save_context",
        disabled=int(context_value) == st.session_state.context_messages,
    ):
        try:
            data = api_request(
                "PUT",
                "/api/config/context",
                json={"context_messages": int(context_value)},
            )
        except ApiError as exc:
            st.error(f"Could not save: {exc}")
        else:
            st.session_state.context_messages = int(data.get("context_messages", context_value))
            st.session_state.settings_version += 1
            push_flash("success", "Context size saved.")
            st.rerun()

    st.divider()

    # ---- workspace ----------------------------------------------
    st.markdown("**Workspace**")
    st.caption("Chats and documents belong to this workspace ID.")

    workspace = st.text_input(
        "Workspace ID",
        value=current_user(),
        key=f"workspace_{version}",
        max_chars=100,
        label_visibility="collapsed",
    ).strip()

    if st.button("Switch workspace", key="switch_workspace", disabled=workspace == current_user()):
        if not USER_ID_PATTERN.match(workspace):
            st.error("Use letters, numbers and . _ @ - only (max 100 characters).")
        else:
            st.session_state.user_id = workspace
            reset_workspace_state()
            st.session_state.pending_view = VIEW_CHAT
            st.rerun()

    st.divider()

    # ---- danger zone --------------------------------------------
    chat_count = len(st.session_state.chats)

    st.markdown("**Delete all chats**")
    st.caption(
        f"Permanently deletes all {plural(chat_count, 'chat')} and their messages in "
        f"workspace “{current_user()}”. Uploaded documents and settings are not affected."
    )

    confirmed = st.checkbox("I understand this cannot be undone", key="confirm_delete_all")

    if st.button(
        "Delete all chats",
        key="delete_all_chats",
        disabled=not confirmed or chat_count == 0,
    ):
        delete_all_chats()
        st.rerun()


# ============================================================
# PAGE
# ============================================================

if st.session_state.pending_view:
    st.session_state.view = st.session_state.pending_view
    st.session_state.pending_view = None

if not st.session_state.chats_loaded:
    load_chats()

    # Resume the latest conversation once per session.
    if (
        st.session_state.chats_loaded
        and not st.session_state.chats_autoopened
        and st.session_state.active_chat_id is None
        and st.session_state.chats
    ):
        st.session_state.active_chat_id = st.session_state.chats[0]["chat_id"]

    if st.session_state.chats_loaded:
        st.session_state.chats_autoopened = True

if not st.session_state.documents_loaded:
    load_documents()

if not st.session_state.settings_loaded:
    load_settings()

search_query = render_sidebar()

view = st.segmented_control(
    "View",
    VIEWS,
    key="view",
    label_visibility="collapsed",
) or VIEW_CHAT

if view == VIEW_RESPONSES:
    render_responses_tab()
elif view == VIEW_SETTINGS:
    render_settings_tab()
else:
    render_chat_tab()
