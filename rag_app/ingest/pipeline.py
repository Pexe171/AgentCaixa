"""Pipeline de ingestão: DOCX -> chunks semânticos -> embeddings -> Qdrant."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

from rag_app.config import AppSettings, load_settings
from rag_app.ingest.parser_docx import Block, parse_docx_to_blocks


@dataclass(frozen=True)
class SemanticChunk:
    text: str
    metadata: dict[str, str]


class OllamaEmbeddingClient:
    """Cliente simples para embeddings via Ollama HTTP API."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self._settings.OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": self._settings.OLLAMA_EMBEDDING_MODEL,
                "prompt": text,
            },
            timeout=self._settings.OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("Resposta inválida do Ollama: embedding ausente.")
        return [float(value) for value in embedding]


class QdrantVectorSink:
    """Persistência de vetores no Qdrant com criação automática da coleção."""

    def __init__(self, settings: AppSettings) -> None:
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:  # pragma: no cover - depende do extra rag
            raise RuntimeError(
                "qdrant-client não está instalado. Use `pip install .[rag]`."
            ) from exc

        self._settings = settings
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

    def ensure_collection(self, vector_size: int) -> None:
        from qdrant_client import models

        collection_name = self._settings.QDRANT_COLLECTION
        exists = self._client.collection_exists(collection_name=collection_name)
        if exists:
            return
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_points(self, points: list[dict[str, object]]) -> None:
        from qdrant_client import models

        self._client.upsert(
            collection_name=self._settings.QDRANT_COLLECTION,
            points=[
                models.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point["payload"],
                )
                for point in points
            ],
            wait=True,
        )


class OllamaQdrantIngestionPipeline:
    def __init__(
        self,
        settings: AppSettings | None = None,
        embedding_client: OllamaEmbeddingClient | None = None,
        vector_sink: QdrantVectorSink | None = None,
    ) -> None:
        self._settings = settings or load_settings()
        self._embedding_client = (
            embedding_client or OllamaEmbeddingClient(self._settings)
        )
        self._vector_sink = vector_sink or QdrantVectorSink(self._settings)

    def _metadata_from_block(self, block: Block) -> dict[str, str]:
        return {
            "source_file": block.file_name,
            "created_at": block.created_at,
            "section": block.section,
            "block_order": str(block.order),
            "block_type": block.type,
        }

    def _build_semantic_chunks(self, file_path: Path) -> list[SemanticChunk]:
        blocks = parse_docx_to_blocks(file_path)
        chunks: list[SemanticChunk] = []
        current_chunk: list[str] = []
        current_meta: dict[str, str] | None = None
        max_chars = self._settings.CHUNK_MAX_CHARS

        def flush_chunk() -> None:
            nonlocal current_chunk, current_meta
            if not current_chunk or current_meta is None:
                return
            chunks.append(
                SemanticChunk(
                    text="\n\n".join(current_chunk),
                    metadata=current_meta,
                )
            )
            current_chunk = []
            current_meta = None

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            metadata = self._metadata_from_block(block)
            should_flush = bool(
                current_meta
                and (
                    current_meta.get("section") != metadata.get("section")
                    or len("\n\n".join(current_chunk + [text])) > max_chars
                )
            )
            if should_flush:
                flush_chunk()
            if current_meta is None:
                current_meta = metadata
            current_chunk.append(text)

        flush_chunk()
        return chunks

    def ingest_docx(self, file_path: Path) -> int:
        semantic_chunks = self._build_semantic_chunks(Path(file_path))
        if not semantic_chunks:
            return 0

        points: list[dict[str, object]] = []
        vector_size = 0
        for index, chunk in enumerate(semantic_chunks, start=1):
            vector = self._embedding_client.embed(chunk.text)
            vector_size = len(vector)
            identity = f"{file_path}:{index}:{chunk.text}"
            stable_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
            points.append(
                {
                    "id": stable_id,
                    "vector": vector,
                    "payload": {
                        **chunk.metadata,
                        "chunk_text": chunk.text,
                        "chunk_order": index,
                        "source_path": str(file_path),
                    },
                }
            )

        self._vector_sink.ensure_collection(vector_size=vector_size)
        self._vector_sink.upsert_points(points)
        return len(points)
