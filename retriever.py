"""Busca híbrida com ChromaDB + BM25 para alta precisão em RAG.

Este módulo oferece:
- indexação local de chunks no ChromaDB usando embeddings do Ollama;
- índice BM25 para busca lexical (siglas, códigos, números de leis);
- busca híbrida que combina ranking vetorial e lexical com estratégia de fusão.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class ResultadoBusca:
    """Representa um item retornado na busca híbrida."""

    id: str
    conteudo: str
    score_hibrido: float
    score_bm25: float
    score_vetorial: float
    metadados: dict[str, Any]


def tokenizar(texto: str) -> list[str]:
    """Tokeniza texto preservando letras, números e separadores úteis."""

    return re.findall(r"[\w\-./]+", texto.lower(), flags=re.UNICODE)


class HybridRetriever:
    """Retriever híbrido com persistência local e foco em precisão."""

    def __init__(
        self,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "documentos",
        ollama_model: str = "nomic-embed-text",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.ollama_model = ollama_model

        self._chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self._embedding_function = OllamaEmbeddingFunction(model_name=ollama_model)
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metas: list[dict[str, Any]] = []
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b

    @property
    def collection(self) -> Collection:
        return self._collection

    def indexar_chunks(self, chunks: list[dict[str, Any]], limpar_colecao: bool = False) -> None:
        """Indexa chunks no ChromaDB e reconstrói o índice BM25.

        Espera uma lista com itens contendo ao menos:
        - id: identificador do chunk
        - conteudo: texto do chunk
        """

        if not chunks:
            raise ValueError("A lista de chunks está vazia.")

        ids: list[str] = []
        documentos: list[str] = []
        metadados: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            conteudo = str(chunk.get("conteudo", "")).strip()
            if not conteudo:
                continue

            chunk_id = str(chunk.get("id", f"chunk-{i+1}"))
            ids.append(chunk_id)
            documentos.append(conteudo)

            meta = {k: v for k, v in chunk.items() if k != "conteudo"}
            if "id" not in meta:
                meta["id"] = chunk_id
            metadados.append(meta)

        if not documentos:
            raise ValueError("Nenhum chunk válido com conteúdo foi fornecido.")

        if limpar_colecao:
            self._recriar_colecao()

        self._collection.upsert(ids=ids, documents=documentos, metadatas=metadados)
        self._reconstruir_bm25()

    def carregar_chunks_do_json(self, caminho_json: Path, limpar_colecao: bool = False) -> None:
        """Carrega chunks no formato da Fase 1 e indexa no sistema híbrido."""

        payload = json.loads(caminho_json.read_text(encoding="utf-8"))
        chunks = payload.get("chunks", [])
        if not isinstance(chunks, list):
            raise ValueError("JSON inválido: campo 'chunks' deve ser uma lista.")

        self.indexar_chunks(chunks=chunks, limpar_colecao=limpar_colecao)

    def buscar(
        self,
        pergunta: str,
        top_k: int = 8,
        peso_bm25: float = 0.65,
        peso_vetorial: float = 0.35,
        k_rrf: int = 60,
    ) -> list[ResultadoBusca]:
        """Executa busca híbrida e retorna os melhores contextos.

        Estratégia:
        1. Busca vetorial no ChromaDB.
        2. Busca lexical BM25.
        3. Fusão por RRF ponderada, favorecendo BM25 para precisão literal.
        """

        if not pergunta or not pergunta.strip():
            raise ValueError("A pergunta não pode ser vazia.")
        if top_k <= 0:
            raise ValueError("top_k deve ser maior que zero.")
        if peso_bm25 < 0 or peso_vetorial < 0:
            raise ValueError("Os pesos não podem ser negativos.")

        if self._bm25 is None:
            self._reconstruir_bm25()

        resultados_vetoriais = self._buscar_vetorial(pergunta, top_k=top_k)
        resultados_bm25 = self._buscar_bm25(pergunta, top_k=top_k)

        ranking_final = self._fundir_rankings(
            resultados_bm25=resultados_bm25,
            resultados_vetoriais=resultados_vetoriais,
            top_k=top_k,
            peso_bm25=peso_bm25,
            peso_vetorial=peso_vetorial,
            k_rrf=k_rrf,
        )

        return ranking_final

    def _recriar_colecao(self) -> None:
        """Remove e recria a coleção para reindexação limpa."""

        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def _reconstruir_bm25(self) -> None:
        """Reconstrói o índice BM25 a partir da coleção armazenada no Chroma."""

        dados = self._collection.get(include=["documents", "metadatas"])
        documentos = dados.get("documents") or []
        ids = dados.get("ids") or []
        metadados = dados.get("metadatas") or []

        self._bm25_docs = [doc for doc in documentos if isinstance(doc, str)]
        self._bm25_ids = [str(doc_id) for doc_id in ids]
        self._bm25_metas = metadados if isinstance(metadados, list) else [{} for _ in self._bm25_docs]

        if not self._bm25_docs:
            self._bm25 = None
            return

        corpus_tokenizado = [tokenizar(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(corpus_tokenizado, k1=self._bm25_k1, b=self._bm25_b)

    def _buscar_vetorial(self, pergunta: str, top_k: int) -> list[dict[str, Any]]:
        """Consulta vetorial no ChromaDB."""

        resposta = self._collection.query(
            query_texts=[pergunta],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = resposta.get("ids", [[]])[0]
        docs = resposta.get("documents", [[]])[0]
        metas = resposta.get("metadatas", [[]])[0]
        dists = resposta.get("distances", [[]])[0]

        itens: list[dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            distancia = float(dists[idx]) if idx < len(dists) else 1.0
            score_similaridade = max(0.0, 1.0 - distancia)
            itens.append(
                {
                    "id": str(chunk_id),
                    "conteudo": docs[idx] if idx < len(docs) else "",
                    "metadados": metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {},
                    "score_vetorial": score_similaridade,
                }
            )

        return itens

    def _buscar_bm25(self, pergunta: str, top_k: int) -> list[dict[str, Any]]:
        """Consulta lexical BM25 com normalização de score."""

        if self._bm25 is None:
            return []

        query_tokens = tokenizar(pergunta)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return []

        max_score = max(float(s) for s in scores) or 1.0
        indices_ordenados = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        resultados: list[dict[str, Any]] = []
        for i in indices_ordenados:
            score_normalizado = float(scores[i]) / max_score if max_score > 0 else 0.0
            resultados.append(
                {
                    "id": self._bm25_ids[i],
                    "conteudo": self._bm25_docs[i],
                    "metadados": self._bm25_metas[i] if i < len(self._bm25_metas) else {},
                    "score_bm25": score_normalizado,
                }
            )

        return resultados

    def _fundir_rankings(
        self,
        resultados_bm25: list[dict[str, Any]],
        resultados_vetoriais: list[dict[str, Any]],
        top_k: int,
        peso_bm25: float,
        peso_vetorial: float,
        k_rrf: int,
    ) -> list[ResultadoBusca]:
        """Combina rankings usando Reciprocal Rank Fusion ponderada."""

        acumulador: dict[str, dict[str, Any]] = {}

        for pos, item in enumerate(resultados_bm25, start=1):
            chunk_id = item["id"]
            entrada = acumulador.setdefault(
                chunk_id,
                {
                    "id": chunk_id,
                    "conteudo": item.get("conteudo", ""),
                    "metadados": item.get("metadados", {}),
                    "score_bm25": item.get("score_bm25", 0.0),
                    "score_vetorial": 0.0,
                    "score_hibrido": 0.0,
                },
            )
            entrada["score_bm25"] = max(float(entrada["score_bm25"]), float(item.get("score_bm25", 0.0)))
            entrada["score_hibrido"] += peso_bm25 * (1.0 / (k_rrf + pos))

        for pos, item in enumerate(resultados_vetoriais, start=1):
            chunk_id = item["id"]
            entrada = acumulador.setdefault(
                chunk_id,
                {
                    "id": chunk_id,
                    "conteudo": item.get("conteudo", ""),
                    "metadados": item.get("metadados", {}),
                    "score_bm25": 0.0,
                    "score_vetorial": item.get("score_vetorial", 0.0),
                    "score_hibrido": 0.0,
                },
            )
            entrada["score_vetorial"] = max(
                float(entrada["score_vetorial"]), float(item.get("score_vetorial", 0.0))
            )
            if not entrada.get("conteudo"):
                entrada["conteudo"] = item.get("conteudo", "")
            if not entrada.get("metadados"):
                entrada["metadados"] = item.get("metadados", {})
            entrada["score_hibrido"] += peso_vetorial * (1.0 / (k_rrf + pos))

        ordenados = sorted(acumulador.values(), key=lambda x: x["score_hibrido"], reverse=True)[:top_k]

        return [
            ResultadoBusca(
                id=item["id"],
                conteudo=item["conteudo"],
                score_hibrido=float(item["score_hibrido"]),
                score_bm25=float(item["score_bm25"]),
                score_vetorial=float(item["score_vetorial"]),
                metadados=item["metadados"],
            )
            for item in ordenados
        ]


def parsear_argumentos() -> argparse.Namespace:
    """CLI para indexação e consulta rápida do retriever híbrido."""

    parser = argparse.ArgumentParser(description="Busca híbrida com ChromaDB + BM25")
    parser.add_argument("--chunks-json", type=Path, help="Caminho para JSON de chunks da Fase 1")
    parser.add_argument("--pergunta", type=str, help="Pergunta para busca")
    parser.add_argument("--top-k", type=int, default=8, help="Quantidade de resultados")
    parser.add_argument("--chroma-dir", type=str, default="./chroma_db", help="Diretório de persistência")
    parser.add_argument("--collection", type=str, default="documentos", help="Nome da coleção")
    parser.add_argument("--modelo", type=str, default="nomic-embed-text", help="Modelo de embedding Ollama")
    parser.add_argument(
        "--limpar",
        action="store_true",
        help="Limpa e recria a coleção antes de indexar",
    )
    return parser.parse_args()


def main() -> None:
    """Execução via terminal para fase de validação rápida."""

    args = parsear_argumentos()
    retriever = HybridRetriever(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        ollama_model=args.modelo,
    )

    if args.chunks_json:
        retriever.carregar_chunks_do_json(args.chunks_json, limpar_colecao=args.limpar)
        print(f"Chunks indexados com sucesso na coleção '{args.collection}'.")

    if args.pergunta:
        resultados = retriever.buscar(args.pergunta, top_k=args.top_k)
        if not resultados:
            print("Nenhum resultado encontrado.")
            return

        for i, item in enumerate(resultados, start=1):
            print(f"\n[{i}] id={item.id}")
            print(
                "scores -> "
                f"hibrido={item.score_hibrido:.6f} | "
                f"bm25={item.score_bm25:.4f} | "
                f"vetorial={item.score_vetorial:.4f}"
            )
            print(item.conteudo[:500])


if __name__ == "__main__":
    main()
