import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import redis

from app.config import settings
from app.services.errors import ConflictError, NotFoundError


DEFAULT_CHAT_TITLE = "New chat"
MAX_CHAT_TITLE_CHARS = 60
MAX_TRANSACTION_RETRIES = 5


def make_chat_title(text: str) -> str:
    """Single-line chat title derived from the first user message."""

    title = " ".join(str(text).split())

    if not title:
        return DEFAULT_CHAT_TITLE

    if len(title) > MAX_CHAT_TITLE_CHARS:
        return title[: MAX_CHAT_TITLE_CHARS - 1].rstrip() + "…"

    return title


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RedisService:

    def __init__(self):
        self.client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            health_check_interval=30,
        )

    # ---------------------------------------------------------
    # Connection
    # ---------------------------------------------------------

    def ping(self) -> bool:
        return bool(self.client.ping())

    # ---------------------------------------------------------
    # Keys
    #
    #   user:{uid}:chats                    sorted set  chat_id -> updated_at
    #   user:{uid}:chat:{cid}:meta          hash        title, created_at, updated_at
    #   user:{uid}:chat:{cid}:messages      list        JSON messages (full history)
    #   user:{uid}:chat                     list        legacy single history (migrated)
    #   user:{uid}:documents                hash        document registry
    #
    # chat_id values are server-generated hex ids. The routes validate
    # their format, and every chat operation first checks the meta key,
    # so a client can never address an arbitrary key.
    # ---------------------------------------------------------

    @staticmethod
    def chat_key(user_id: str) -> str:
        """Legacy single-history key (kept only for migration)."""
        return f"user:{user_id}:chat"

    @staticmethod
    def chat_index_key(user_id: str) -> str:
        return f"user:{user_id}:chats"

    @staticmethod
    def chat_meta_key(user_id: str, chat_id: str) -> str:
        return f"user:{user_id}:chat:{chat_id}:meta"

    @staticmethod
    def chat_messages_key(user_id: str, chat_id: str) -> str:
        return f"user:{user_id}:chat:{chat_id}:messages"

    @staticmethod
    def document_key(user_id: str) -> str:
        return f"user:{user_id}:documents"

    # ---------------------------------------------------------
    # Message helpers
    # ---------------------------------------------------------

    @staticmethod
    def _decode_message(raw: str, index: int) -> dict[str, Any]:
        message = json.loads(raw)

        # Messages written before ids existed get a stable positional id.
        message.setdefault("message_id", f"legacy-{index}")

        return message

    @staticmethod
    def _summary(
        chat_id: str,
        meta: dict[str, str],
        message_count: int,
    ) -> dict[str, Any]:

        return {
            "chat_id": chat_id,
            "title": meta.get("title") or DEFAULT_CHAT_TITLE,
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "message_count": message_count,
        }

    def _require_chat(self, user_id: str, chat_id: str) -> None:
        if not self.client.exists(self.chat_meta_key(user_id, chat_id)):
            raise NotFoundError("Chat not found.")

    # ---------------------------------------------------------
    # Chats: create / list / read
    # ---------------------------------------------------------

    def create_chat(
        self,
        user_id: str,
        title: str | None = None,
    ) -> dict[str, Any]:

        chat_id = uuid.uuid4().hex
        now = _now()

        meta = {
            "title": make_chat_title(title) if title else DEFAULT_CHAT_TITLE,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        pipe = self.client.pipeline()
        pipe.hset(self.chat_meta_key(user_id, chat_id), mapping=meta)
        pipe.zadd(self.chat_index_key(user_id), {chat_id: now.timestamp()})
        pipe.execute()

        return self._summary(chat_id, meta, 0)

    def list_chats(self, user_id: str) -> list[dict[str, Any]]:
        """All chats of a user, most recently updated first."""

        self._migrate_legacy_history(user_id)

        chat_ids = self.client.zrevrange(self.chat_index_key(user_id), 0, -1)

        if not chat_ids:
            return []

        pipe = self.client.pipeline(transaction=False)

        for chat_id in chat_ids:
            pipe.hgetall(self.chat_meta_key(user_id, chat_id))
            pipe.llen(self.chat_messages_key(user_id, chat_id))

        results = pipe.execute()

        chats = []

        for position, chat_id in enumerate(chat_ids):
            meta = results[position * 2]
            count = results[position * 2 + 1]

            if not meta:  # index entry without data: ignore
                continue

            chats.append(self._summary(chat_id, meta, int(count)))

        return chats

    def get_chat(self, user_id: str, chat_id: str) -> dict[str, Any]:

        meta = self.client.hgetall(self.chat_meta_key(user_id, chat_id))

        if not meta:
            raise NotFoundError("Chat not found.")

        count = self.client.llen(self.chat_messages_key(user_id, chat_id))

        return self._summary(chat_id, meta, int(count))

    def get_messages(self, user_id: str, chat_id: str) -> list[dict[str, Any]]:
        """The complete, unbounded history of one chat."""

        self._require_chat(user_id, chat_id)

        raw_items = self.client.lrange(
            self.chat_messages_key(user_id, chat_id), 0, -1
        )

        return [
            self._decode_message(raw, index)
            for index, raw in enumerate(raw_items)
        ]

    def get_recent_history(
        self,
        user_id: str,
        chat_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Only the latest `limit` messages: what the LLM is allowed to see."""

        if limit <= 0:
            return []

        key = self.chat_messages_key(user_id, chat_id)
        total = self.client.llen(key)

        raw_items = self.client.lrange(key, -limit, -1)

        first_index = max(total - len(raw_items), 0)

        return [
            self._decode_message(raw, first_index + offset)
            for offset, raw in enumerate(raw_items)
        ]

    def count_messages(self, user_id: str, chat_id: str) -> int:
        return int(self.client.llen(self.chat_messages_key(user_id, chat_id)))

    # ---------------------------------------------------------
    # Chats: write
    # ---------------------------------------------------------

    def add_messages(
        self,
        user_id: str,
        chat_id: str,
        messages: list[dict[str, Any]],
        title_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Append messages atomically; fails if the chat was deleted meanwhile."""

        now = _now()

        prepared = []

        for message in messages:
            item = dict(message)
            item.setdefault("message_id", uuid.uuid4().hex)
            item.setdefault("timestamp", now.isoformat())
            prepared.append(item)

        meta_key = self.chat_meta_key(user_id, chat_id)
        messages_key = self.chat_messages_key(user_id, chat_id)

        for _ in range(MAX_TRANSACTION_RETRIES):
            with self.client.pipeline() as pipe:
                try:
                    pipe.watch(meta_key)

                    if not pipe.exists(meta_key):
                        raise NotFoundError("Chat not found.")

                    current_title = pipe.hget(meta_key, "title")

                    fields = {"updated_at": now.isoformat()}

                    if title_hint and current_title == DEFAULT_CHAT_TITLE:
                        fields["title"] = title_hint

                    pipe.multi()
                    pipe.rpush(messages_key, *[json.dumps(item) for item in prepared])
                    pipe.hset(meta_key, mapping=fields)
                    pipe.zadd(self.chat_index_key(user_id), {chat_id: now.timestamp()})
                    pipe.execute()

                    return prepared

                except redis.WatchError:
                    continue

        raise ConflictError("The chat was modified concurrently. Please retry.")

    def clear_chat(self, user_id: str, chat_id: str) -> None:
        """Remove every message of ONE chat. The chat itself stays."""

        self._require_chat(user_id, chat_id)

        now = _now()

        pipe = self.client.pipeline()
        pipe.delete(self.chat_messages_key(user_id, chat_id))
        pipe.hset(
            self.chat_meta_key(user_id, chat_id),
            mapping={"title": DEFAULT_CHAT_TITLE, "updated_at": now.isoformat()},
        )
        pipe.zadd(self.chat_index_key(user_id), {chat_id: now.timestamp()})
        pipe.execute()

    def delete_chat(self, user_id: str, chat_id: str) -> None:
        """Remove ONE chat completely (messages, metadata, sidebar entry)."""

        self._require_chat(user_id, chat_id)

        pipe = self.client.pipeline()
        pipe.delete(self.chat_meta_key(user_id, chat_id))
        pipe.delete(self.chat_messages_key(user_id, chat_id))
        pipe.zrem(self.chat_index_key(user_id), chat_id)
        pipe.execute()

    def delete_all_chats(self, user_id: str) -> int:
        """Remove every chat of a user. Documents are not touched."""

        chat_ids = self.client.zrange(self.chat_index_key(user_id), 0, -1)

        pipe = self.client.pipeline()

        for chat_id in chat_ids:
            pipe.delete(self.chat_meta_key(user_id, chat_id))
            pipe.delete(self.chat_messages_key(user_id, chat_id))

        pipe.delete(self.chat_index_key(user_id))
        pipe.delete(self.chat_key(user_id))  # un-migrated legacy history
        pipe.execute()

        return len(chat_ids)

    # ---------------------------------------------------------
    # Responses: edit / restore
    # ---------------------------------------------------------

    def _mutate_message(
        self,
        user_id: str,
        chat_id: str,
        message_id: str,
        mutate: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Optimistic read-modify-write of one message inside the list."""

        meta_key = self.chat_meta_key(user_id, chat_id)
        messages_key = self.chat_messages_key(user_id, chat_id)

        for _ in range(MAX_TRANSACTION_RETRIES):
            with self.client.pipeline() as pipe:
                try:
                    pipe.watch(meta_key, messages_key)

                    if not pipe.exists(meta_key):
                        raise NotFoundError("Chat not found.")

                    raw_items = pipe.lrange(messages_key, 0, -1)

                    target = None

                    for index, raw in enumerate(raw_items):
                        message = self._decode_message(raw, index)

                        if message["message_id"] == message_id:
                            target = (index, message)
                            break

                    if target is None:
                        raise NotFoundError("Message not found.")

                    index, message = target
                    updated = mutate(message)

                    pipe.multi()
                    pipe.lset(messages_key, index, json.dumps(updated))
                    pipe.execute()

                    return updated

                except redis.WatchError:
                    continue

        raise ConflictError("The chat was modified concurrently. Please retry.")

    def update_response(
        self,
        user_id: str,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Replace an assistant response; the original text is kept."""

        def mutate(message: dict[str, Any]) -> dict[str, Any]:
            if message.get("role") != "assistant":
                raise ValueError("Only assistant responses can be edited.")

            original = message.get("original_content", message.get("content", ""))

            if content == original:
                for field in ("edited", "edited_at", "original_content"):
                    message.pop(field, None)
                message["content"] = original
                return message

            message.update(
                content=content,
                edited=True,
                edited_at=_now().isoformat(),
                original_content=original,
            )

            return message

        return self._mutate_message(user_id, chat_id, message_id, mutate)

    def restore_response(
        self,
        user_id: str,
        chat_id: str,
        message_id: str,
    ) -> dict[str, Any]:
        """Undo an edit (no-op if the response was never edited)."""

        def mutate(message: dict[str, Any]) -> dict[str, Any]:
            if "original_content" in message:
                message["content"] = message["original_content"]

            for field in ("edited", "edited_at", "original_content"):
                message.pop(field, None)

            return message

        return self._mutate_message(user_id, chat_id, message_id, mutate)

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search_chats(
        self,
        user_id: str,
        query: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Chats whose title or any message contains `query` (case-insensitive)."""

        needle = query.strip().lower()

        if not needle:
            return []

        results = []

        for chat in self.list_chats(user_id):
            snippet = None

            if needle in chat["title"].lower():
                snippet = chat["title"]
            else:
                raw_items = self.client.lrange(
                    self.chat_messages_key(user_id, chat["chat_id"]), 0, -1
                )

                for raw in raw_items:
                    content = str(json.loads(raw).get("content", ""))
                    position = content.lower().find(needle)

                    if position >= 0:
                        start = max(position - 40, 0)
                        snippet = content[start : position + len(needle) + 40]
                        snippet = " ".join(snippet.split())
                        break

            if snippet is not None:
                results.append({**chat, "snippet": snippet})

                if len(results) >= limit:
                    break

        return results

    # ---------------------------------------------------------
    # Legacy migration
    #
    # Older versions kept ONE list per user in `user:{uid}:chat`.
    # It is moved (atomically, via RENAME) into a regular chat the
    # first time chats are listed, so no history is lost.
    # ---------------------------------------------------------

    def _migrate_legacy_history(self, user_id: str) -> None:
        legacy_key = self.chat_key(user_id)

        if not self.client.exists(legacy_key):
            return

        chat_id = uuid.uuid4().hex
        now = _now()

        with self.client.pipeline() as pipe:
            try:
                pipe.watch(legacy_key)

                if not pipe.exists(legacy_key):
                    return

                first_raw = pipe.lindex(legacy_key, 0)
                last_raw = pipe.lindex(legacy_key, -1)

                title = DEFAULT_CHAT_TITLE
                created_at = updated_at = now.isoformat()

                try:
                    first = json.loads(first_raw)
                    last = json.loads(last_raw)
                    created_at = first.get("timestamp") or created_at
                    updated_at = last.get("timestamp") or updated_at

                    if first.get("role") == "user":
                        title = make_chat_title(first.get("content", ""))
                except (TypeError, ValueError):
                    pass

                try:
                    score = datetime.fromisoformat(updated_at).timestamp()
                except ValueError:
                    score = now.timestamp()

                pipe.multi()
                pipe.rename(legacy_key, self.chat_messages_key(user_id, chat_id))
                pipe.hset(
                    self.chat_meta_key(user_id, chat_id),
                    mapping={
                        "title": title,
                        "created_at": created_at,
                        "updated_at": updated_at,
                    },
                )
                pipe.zadd(self.chat_index_key(user_id), {chat_id: score})
                pipe.execute()

            except redis.WatchError:
                return  # someone else migrated or wrote; next call retries

    # ---------------------------------------------------------
    # Document registry
    # ---------------------------------------------------------

    def save_document(
        self,
        user_id: str,
        file_id: str,
        document: dict[str, Any],
    ) -> None:

        self.client.hset(
            self.document_key(user_id),
            file_id,
            json.dumps(document),
        )

    def get_document(
        self,
        user_id: str,
        file_id: str,
    ) -> dict[str, Any] | None:

        value = self.client.hget(
            self.document_key(user_id),
            file_id,
        )

        if value is None:
            return None

        return json.loads(value)

    def get_documents(
        self,
        user_id: str,
    ) -> list[dict[str, Any]]:

        values = self.client.hvals(
            self.document_key(user_id)
        )

        return [
            json.loads(value)
            for value in values
        ]

    def delete_document(
        self,
        user_id: str,
        file_id: str,
    ) -> None:

        self.client.hdel(
            self.document_key(user_id),
            file_id,
        )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    def get_prompt(self) -> str | None:

        return self.client.get(
            "config:system_prompt"
        )

    def set_prompt(self, prompt: str) -> None:

        self.client.set(
            "config:system_prompt",
            prompt,
        )

    def get_context_limit(self) -> int | None:

        value = self.client.get(
            "config:chat_context_messages"
        )

        if value is None:
            return None

        return int(value)

    def set_context_limit(self, value: int) -> None:

        self.client.set(
            "config:chat_context_messages",
            value,
        )


redis_service = RedisService()
