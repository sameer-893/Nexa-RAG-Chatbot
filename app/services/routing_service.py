from typing import Literal, Any

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from app.config import settings


class RouteDecision(BaseModel):
    """
    Decision made by the agentic router.
    """

    route: Literal[
        "direct",
        "rag",
        "web",
    ]

    reason: str


class RoutingService:

    def __init__(self):

        # =====================================================
        # Router LLM
        # =====================================================

        self.router_llm = ChatOpenAI(
            model=settings.router_model,
            api_key=settings.openai_api_key,
        )

        self.structured_router = (
            self.router_llm.with_structured_output(
                RouteDecision,
                method="json_schema",
                strict=True,
            )
        )

        # =====================================================
        # Direct-answer LLM
        # =====================================================

        self.answer_llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
        )

        # =====================================================
        # Web-search LLM
        #
        # OpenAI server-side web search
        # =====================================================

        self.web_llm = ChatOpenAI(
            model=settings.web_search_model,
            api_key=settings.openai_api_key,
            output_version="responses/v1",
        )

        self.web_llm_with_search = (
            self.web_llm.bind_tools(
                [
                    {
                        "type": "web_search"
                    }
                ],
                tool_choice="required",
            )
        )

    # =========================================================
    # Convert Redis history → LangChain messages
    # =========================================================

    @staticmethod
    def history_to_messages(
        history: list[dict[str, Any]],
    ) -> list:

        messages = []

        for item in history:

            role = item.get("role")
            content = item.get("content", "")

            if not content:
                continue

            if role == "user":

                messages.append(
                    HumanMessage(
                        content=content
                    )
                )

            elif role == "assistant":

                messages.append(
                    AIMessage(
                        content=content
                    )
                )

        return messages

    # =========================================================
    # AGENTIC ROUTER
    # =========================================================

    def decide_route(
        self,
        question: str,
        history: list[dict[str, Any]],
        document_names: list[str],
    ) -> RouteDecision:

        history_messages = (
            self.history_to_messages(
                history
            )
        )

        available_documents = (
            ", ".join(document_names)
            if document_names
            else "No documents uploaded."
        )

        router_system = f"""
You are an intelligent routing agent for a RAG chatbot.

You must choose exactly ONE route:

1. "direct"
   Use the normal LLM without retrieval or web search.

   Choose this for:
   - General explanations
   - Coding questions
   - Writing
   - Reasoning
   - Math
   - Stable/general knowledge
   - Questions that do not require external documents

2. "rag"
   Use the user's uploaded documents through Pinecone.

   Choose this when:
   - The user asks about uploaded files
   - The user explicitly says "according to my document"
   - The question refers to lecture notes, PDFs, files, etc.
   - The answer should come from the user's private knowledge base

3. "web"
   Use internet search.

   Choose this when:
   - The question asks for current/latest/recent information
   - News
   - Current people/companies/events
   - Current prices
   - Current software/library information
   - The answer requires information from the internet
   - The user explicitly asks to search online

Available user documents:
{available_documents}

Important:

- Use conversation history to understand references such as
  "that file", "the previous answer", etc.
- Prefer RAG when the answer is specifically about uploaded documents.
- Prefer web search for information that can change over time.
- Prefer direct for normal stable questions.
- Return exactly one route.
"""

        messages = [
            SystemMessage(
                content=router_system
            )
        ]

        messages.extend(
            history_messages
        )

        messages.append(
            HumanMessage(
                content=question
            )
        )

        result = (
            self.structured_router.invoke(
                messages
            )
        )

        return result

    # =========================================================
    # DIRECT ANSWER
    # =========================================================

    def direct_answer(
        self,
        question: str,
        history: list[dict[str, Any]],
        system_prompt: str,
    ) -> str:

        messages = [
            SystemMessage(
                content=system_prompt
            )
        ]

        messages.extend(
            self.history_to_messages(
                history
            )
        )

        messages.append(
            HumanMessage(
                content=question
            )
        )

        response = self.answer_llm.invoke(
            messages
        )

        return self.extract_text(
            response
        )

    # =========================================================
    # WEB ANSWER
    # =========================================================

    def web_answer(
        self,
        question: str,
        history: list[dict[str, Any]],
        system_prompt: str,
    ) -> dict[str, Any]:

        web_system_prompt = f"""
{system_prompt}

You are currently answering using internet search.

Rules:

1. Search the web before answering.
2. Prefer reliable and primary sources.
3. For current information, verify the information from the
   search results.
4. Do not invent facts.
5. Give a concise, useful answer.
6. The response may contain URL citations generated by the
   web search tool.
"""

        messages = [
            SystemMessage(
                content=web_system_prompt
            )
        ]

        messages.extend(
            self.history_to_messages(
                history
            )
        )

        messages.append(
            HumanMessage(
                content=question
            )
        )

        response = (
            self.web_llm_with_search.invoke(
                messages
            )
        )

        answer, sources = (
            self.extract_web_response(
                response
            )
        )

        return {
            "answer": answer,
            "sources": sources,
        }

    # =========================================================
    # Response text helper
    # =========================================================

    @staticmethod
    def extract_text(
        response,
    ) -> str:

        content = response.content

        if isinstance(
            content,
            str,
        ):
            return content

        if isinstance(
            content,
            list,
        ):

            parts = []

            for block in content:

                if isinstance(
                    block,
                    str,
                ):
                    parts.append(block)

                elif isinstance(
                    block,
                    dict,
                ):

                    if (
                        block.get("type")
                        == "text"
                    ):
                        parts.append(
                            block.get(
                                "text",
                                "",
                            )
                        )

            return "\n".join(parts).strip()

        return str(content)

    # =========================================================
    # Extract Web Search Citations
    # =========================================================

    @classmethod
    def extract_web_response(
        cls,
        response,
    ) -> tuple[
        str,
        list[dict[str, str]],
    ]:

        content = response.content

        if isinstance(
            content,
            str,
        ):
            return content, []

        text_parts = []
        sources = []

        if not isinstance(
            content,
            list,
        ):

            return (
                str(content),
                [],
            )

        for block in content:

            if not isinstance(
                block,
                dict,
            ):
                continue

            if block.get("type") == "text":

                text = block.get(
                    "text",
                    "",
                )

                if text:
                    text_parts.append(
                        text
                    )

                annotations = block.get(
                    "annotations",
                    [],
                )

                for annotation in annotations:

                    if (
                        annotation.get(
                            "type"
                        )
                        == "url_citation"
                    ):

                        url = annotation.get(
                            "url"
                        )

                        title = annotation.get(
                            "title",
                            url,
                        )

                        if url:

                            sources.append(
                                {
                                    "title": title,
                                    "url": url,
                                }
                            )

        # Remove duplicate URLs
        unique_sources = []
        seen = set()

        for source in sources:

            url = source["url"]

            if url in seen:
                continue

            seen.add(url)
            unique_sources.append(
                source
            )

        return (
            "\n".join(
                text_parts
            ).strip(),
            unique_sources,
        )


routing_service = RoutingService()