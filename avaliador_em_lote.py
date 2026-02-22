"""Fase 5: avaliador em lote para perguntas e respostas com RAG.

Lê perguntas de um arquivo texto, recupera contexto com o retriever híbrido
(Top-K=4 por padrão), consulta provedor local (Ollama) ou cloud (OpenAI)
e salva um relatório CSV para auditoria.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from agent import ErroOllama, ErroOpenAI, responder_com_ollama, responder_com_openai
from query_rewriter import expandir_pergunta
from retriever import HybridRetriever, ResultadoBusca


COLUNAS_RELATORIO = [
    "Data/Hora",
    "Pergunta",
    "Trechos Utilizados",
    "Resposta do Assistente",
    "Avaliação Manual",
]


_LOCK_LOG = Lock()


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


def _responder_com_tolerancia(
    pergunta: str,
    retriever: HybridRetriever,
    top_k: int,
    modelo_llm: str,
    ollama_url: str,
    provedor: str,
) -> dict[str, Any]:
    """Executa uma pergunta e nunca lança erro fatal para o lote."""

    pergunta_tecnica = expandir_pergunta(pergunta, provedor="openai" if provedor == "openai" else "local")
    documentos = retriever.buscar(pergunta=pergunta_tecnica, top_k=top_k)
    trecho_resumo = resumir_trechos(documentos)

    try:
        if provedor == "openai":
            resposta = responder_com_openai(
                documentos=documentos,
                pergunta=pergunta,
                modelo=modelo_llm,
            )
        else:
            resposta = responder_com_ollama(
                documentos=documentos,
                pergunta=pergunta,
                modelo=modelo_llm,
                base_url=ollama_url,
            )
    except (ValueError, ErroOllama, ErroOpenAI, RuntimeError) as erro:
        resposta = f"[Falha ao gerar resposta: {erro}]"

    return {
        "Data/Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Pergunta": pergunta,
        "Trechos Utilizados": trecho_resumo,
        "Resposta do Assistente": resposta,
        "Avaliação Manual": "",
    }


def avaliar_em_lote(
    perguntas: list[str],
    retriever: HybridRetriever,
    top_k: int,
    modelo_llm: str,
    ollama_url: str,
    provedor: str,
    threads: int,
) -> list[dict[str, Any]]:
    """Executa recuperação + geração para um conjunto de perguntas."""

    total = len(perguntas)

    if provedor == "openai":
        resultados_por_indice: dict[int, dict[str, Any]] = {}
        concluidas = 0

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futuros = {
                executor.submit(
                    _responder_com_tolerancia,
                    pergunta,
                    retriever,
                    top_k,
                    modelo_llm,
                    ollama_url,
                    provedor,
                ): indice
                for indice, pergunta in enumerate(perguntas)
            }

            for futuro in as_completed(futuros):
                indice = futuros[futuro]
                resultados_por_indice[indice] = futuro.result()
                concluidas += 1
                with _LOCK_LOG:
                    print(f"Progresso OpenAI: {concluidas}/{total} perguntas processadas.")

        return [resultados_por_indice[i] for i in range(total)]

    linhas_relatorio: list[dict[str, Any]] = []
    for indice, pergunta in enumerate(perguntas, start=1):
        print(f"Respondendo pergunta {indice} de {total}...")
        linhas_relatorio.append(
            _responder_com_tolerancia(
                pergunta=pergunta,
                retriever=retriever,
                top_k=top_k,
                modelo_llm=modelo_llm,
                ollama_url=ollama_url,
                provedor=provedor,
            )
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
        default=None,
        help="Caminho do CSV de saída (padrão dinâmico por provedor)",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Quantidade de trechos por pergunta")
    parser.add_argument("--modelo-llm", type=str, default="llama3", help="Modelo de geração")
    parser.add_argument(
        "--modelo-embedding",
        type=str,
        default="nomic-embed-text",
        help="Modelo de embedding do retriever",
    )
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL base do Ollama")
    parser.add_argument("--chroma-dir", type=str, default="./chroma_db", help="Diretório persistente do ChromaDB")
    parser.add_argument("--collection", type=str, default="documentos", help="Nome da coleção no ChromaDB")
    parser.add_argument(
        "--provedor",
        type=str,
        choices=["local", "openai"],
        default="local",
        help="Provedor de geração: local (Ollama) ou openai",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=50,
        help="Total de threads para modo OpenAI (padrão: 50)",
    )
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

    saida_padrao = Path("relatorio_ouro_openai.csv") if args.provedor == "openai" else Path("relatorio_avaliacao.csv")
    saida_csv = args.saida_csv or saida_padrao

    linhas_relatorio = avaliar_em_lote(
        perguntas=perguntas,
        retriever=retriever,
        top_k=args.top_k,
        modelo_llm=args.modelo_llm,
        ollama_url=args.ollama_url,
        provedor=args.provedor,
        threads=args.threads,
    )

    df = pd.DataFrame(linhas_relatorio, columns=COLUNAS_RELATORIO)
    df.to_csv(saida_csv, index=False, encoding="utf-8-sig")
    print(f"\nRelatório salvo com sucesso em: {saida_csv}")


if __name__ == "__main__":
    main()
