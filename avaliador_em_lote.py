"""Fase 5: avaliador em lote para perguntas e respostas com RAG.

Lê perguntas de um arquivo texto, recupera contexto com o retriever híbrido
(Top-K=4 por padrão), consulta o Ollama e salva um relatório CSV para auditoria.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agent import responder_com_ollama
from retriever import HybridRetriever, ResultadoBusca


COLUNAS_RELATORIO = [
    "Data/Hora",
    "Pergunta",
    "Trechos Utilizados",
    "Resposta do Assistente",
    "Avaliação Manual",
]


def ler_perguntas(caminho_perguntas: Path) -> list[str]:
    """Lê perguntas de um arquivo TXT (uma por linha)."""

    if not caminho_perguntas.exists():
        raise FileNotFoundError(f"Arquivo de perguntas não encontrado: {caminho_perguntas}")

    linhas = caminho_perguntas.read_text(encoding="utf-8").splitlines()
    perguntas = [linha.strip() for linha in linhas if linha.strip()]

    if not perguntas:
        raise ValueError("O arquivo de perguntas está vazio ou só possui linhas em branco.")

    return perguntas


def resumir_trechos(resultados: list[ResultadoBusca], limite_chars: int = 140) -> str:
    """Retorna um resumo curto dos trechos usados para auditoria.

    Formato por item: "<id>: <início do conteúdo>".
    """

    resumos: list[str] = []
    for item in resultados:
        conteudo_curto = item.conteudo.replace("\n", " ").strip()[:limite_chars]
        resumos.append(f"{item.id}: {conteudo_curto}")
    return " | ".join(resumos)


def avaliar_em_lote(
    perguntas: list[str],
    retriever: HybridRetriever,
    top_k: int,
    modelo_llm: str,
    ollama_url: str,
) -> list[dict[str, Any]]:
    """Executa recuperação + geração para um conjunto de perguntas."""

    total = len(perguntas)
    linhas_relatorio: list[dict[str, Any]] = []

    for indice, pergunta in enumerate(perguntas, start=1):
        print(f"Respondendo pergunta {indice} de {total}...")

        documentos = retriever.buscar(pergunta=pergunta, top_k=top_k)
        resposta = responder_com_ollama(
            documentos=documentos,
            pergunta=pergunta,
            modelo=modelo_llm,
            base_url=ollama_url,
        )

        linhas_relatorio.append(
            {
                "Data/Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Pergunta": pergunta,
                "Trechos Utilizados": resumir_trechos(documentos),
                "Resposta do Assistente": resposta,
                "Avaliação Manual": "",
            }
        )

    return linhas_relatorio


def parsear_argumentos() -> argparse.Namespace:
    """Define argumentos da CLI do avaliador em lote."""

    parser = argparse.ArgumentParser(description="Fase 5: avaliador em lote")
    parser.add_argument(
        "--arquivo-perguntas",
        type=Path,
        default=Path("perguntas.txt"),
        help="Arquivo TXT com uma pergunta por linha (padrão: perguntas.txt)",
    )
    parser.add_argument(
        "--saida-csv",
        type=Path,
        default=Path("relatorio_avaliacao.csv"),
        help="Caminho do CSV de saída (padrão: relatorio_avaliacao.csv)",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Quantidade de trechos por pergunta")
    parser.add_argument("--modelo-llm", type=str, default="llama3", help="Modelo de geração do Ollama")
    parser.add_argument(
        "--modelo-embedding",
        type=str,
        default="nomic-embed-text",
        help="Modelo de embedding do retriever",
    )
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL base do Ollama")
    parser.add_argument("--chroma-dir", type=str, default="./chroma_db", help="Diretório persistente do ChromaDB")
    parser.add_argument("--collection", type=str, default="documentos", help="Nome da coleção no ChromaDB")
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada da Fase 5 (avaliação em lote)."""

    args = parsear_argumentos()
    perguntas = ler_perguntas(args.arquivo_perguntas)

    retriever = HybridRetriever(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        ollama_model=args.modelo_embedding,
    )

    linhas_relatorio = avaliar_em_lote(
        perguntas=perguntas,
        retriever=retriever,
        top_k=args.top_k,
        modelo_llm=args.modelo_llm,
        ollama_url=args.ollama_url,
    )

    df = pd.DataFrame(linhas_relatorio, columns=COLUNAS_RELATORIO)
    df.to_csv(args.saida_csv, index=False, encoding="utf-8-sig")
    print(f"\nRelatório salvo com sucesso em: {args.saida_csv}")


if __name__ == "__main__":
    main()
