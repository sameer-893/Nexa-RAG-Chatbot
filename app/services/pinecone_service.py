import time
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.config import settings


class PineconeService:

    def __init__(self):

        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        self.pc = Pinecone(
            api_key=settings.pinecone_api_key
        )

        self._ensure_index()

        self.index = self.pc.Index(
            settings.pinecone_index
        )

        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace=settings.pinecone_namespace,
        )

    # ---------------------------------------------------------
    # Index creation
    # ---------------------------------------------------------

    def _ensure_index(self) -> None:

        existing_indexes = [
            item["name"]
            for item in self.pc.list_indexes()
        ]

        if settings.pinecone_index in existing_indexes:
            return

        self.pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

        # Wait for index to become ready
        while True:

            description = self.pc.describe_index(
                settings.pinecone_index
            )

            ready = False

            try:
                ready = bool(
                    description.status["ready"]
                )
            except (KeyError, TypeError):
                try:
                    ready = bool(
                        description.status.ready
                    )
                except AttributeError:
                    pass

            if ready:
                break

            time.sleep(1)

    # ---------------------------------------------------------
    # Add documents
    # ---------------------------------------------------------

    def add_documents(
        self,
        documents: list,
        ids: list[str],
    ) -> None:

        self.vector_store.add_documents(
            documents,
            ids=ids,
        )

    # ---------------------------------------------------------
    # Similarity search
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int,
    ):

        return self.vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter={
                "user_id": {
                    "$eq": user_id
                }
            },
        )

    # ---------------------------------------------------------
    # Delete a complete file
    #
    # Preferred path: delete by the deterministic vector ids
    # ("{file_id}:{chunk_index}"). It works on every index type.
    # Pinecone accepts at most 1000 ids per delete call.
    # Fallback (ids unknown): delete by metadata filter.
    # ---------------------------------------------------------

    DELETE_BATCH_SIZE = 1000

    def delete_file(
        self,
        user_id: str,
        file_id: str,
        vector_ids: list[str] | None = None,
    ) -> None:

        if vector_ids:

            for start in range(0, len(vector_ids), self.DELETE_BATCH_SIZE):

                self.index.delete(
                    ids=vector_ids[start:start + self.DELETE_BATCH_SIZE],
                    namespace=settings.pinecone_namespace,
                )

            return

        self.index.delete(
            namespace=settings.pinecone_namespace,
            filter={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"file_id": {"$eq": file_id}},
                ]
            },
        )

    # ---------------------------------------------------------
    # Fetch vectors by IDs
    # ---------------------------------------------------------

    def fetch_vectors(
        self,
        ids: list[str],
    ) -> Any:

        return self.index.fetch(
            ids=ids,
            namespace=settings.pinecone_namespace,
        )

    # ---------------------------------------------------------
    # Namespace statistics
    # ---------------------------------------------------------

    def stats(self) -> Any:

        return self.index.describe_index_stats()


pinecone_service = PineconeService()