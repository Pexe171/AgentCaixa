"""Fase 4: Interface Streamlit com chat humanizado e coleta de feedback."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from agent import ErroOllama, responder_com_ollama
from retriever import HybridRetriever

DB_PATH = Path("feedback.db")


def inicializar_banco() -> None:
    """Cria a base de feedback local, se ainda não existir."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_hora TEXT NOT NULL,
                pergunta TEXT NOT NULL,
                resposta TEXT NOT NULL,
                feedback INTEGER NOT NULL CHECK (feedback IN (0, 1)),
                message_id TEXT UNIQUE NOT NULL
            )
            """
        )
        conn.commit()


def salvar_feedback(message_id: str, pergunta: str, resposta: str, valor_feedback: int) -> None:
    """Salva/atualiza o feedback de uma resposta do bot."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO feedback (data_hora, pergunta, resposta, feedback, message_id)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(message_id)
            DO UPDATE SET
                data_hora = excluded.data_hora,
                pergunta = excluded.pergunta,
                resposta = excluded.resposta,
                feedback = excluded.feedback
            """,
            (datetime.now().isoformat(timespec="seconds"), pergunta, resposta, valor_feedback, message_id),
        )
        conn.commit()


def carregar_aprendizado() -> pd.DataFrame:
    """Retorna dados consolidados por data para o gráfico de aprendizado."""

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT data_hora, feedback FROM feedback", conn)

    if df.empty:
        return pd.DataFrame(columns=["data", "taxa_acerto"])

    df["data"] = pd.to_datetime(df["data_hora"]).dt.date
    consolidado = (
        df.groupby("data", as_index=False)["feedback"]
        .mean()
        .rename(columns={"feedback": "taxa_acerto"})
    )
    consolidado["taxa_acerto"] = consolidado["taxa_acerto"] * 100
    return consolidado


@st.cache_resource(show_spinner=False)
def obter_retriever(chroma_dir: str, collection_name: str, modelo_embedding: str) -> HybridRetriever:
    """Inicializa o retriever uma vez por sessão."""

    return HybridRetriever(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        ollama_model=modelo_embedding,
    )


def gerar_resposta(pergunta: str, retriever: HybridRetriever, top_k: int, modelo_llm: str, ollama_url: str) -> str:
    """Executa pipeline de recuperação + resposta final."""

    documentos = retriever.buscar(pergunta, top_k=top_k)
    documentos_dict = [asdict(item) for item in documentos]

    return responder_com_ollama(
        documentos=documentos_dict,
        pergunta=pergunta,
        modelo=modelo_llm,
        base_url=ollama_url,
    )


def renderizar_sidebar() -> None:
    """Exibe o painel lateral com evolução de aprendizado."""

    st.sidebar.header("Gráfico de Aprendizado")
    consolidado = carregar_aprendizado()

    if consolidado.empty:
        st.sidebar.info("Ainda não há feedbacks registrados.")
        return

    taxa_media = consolidado["taxa_acerto"].mean()
    st.sidebar.metric("Taxa média de acerto", f"{taxa_media:.1f}%")
    st.sidebar.line_chart(consolidado.set_index("data")["taxa_acerto"])


def main() -> None:
    """UI principal do chat em Streamlit."""

    st.set_page_config(page_title="Assistente de Documentos", page_icon="💬", layout="wide")
    st.title("💬 Assistente de Documentos")
    st.caption("Chat humanizado com feedback contínuo para evolução da qualidade das respostas.")

    inicializar_banco()
    renderizar_sidebar()

    with st.sidebar.expander("Configurações", expanded=True):
        chroma_dir = st.text_input("Diretório ChromaDB", value="./chroma_db")
        collection_name = st.text_input("Coleção", value="documentos")
        modelo_embedding = st.text_input("Modelo de embedding", value="nomic-embed-text")
        modelo_llm = st.text_input("Modelo LLM", value="llama3")
        ollama_url = st.text_input("URL do Ollama", value="http://localhost:11434")
        top_k = st.slider("Top-K de contexto", min_value=1, max_value=15, value=4)

    if "historico" not in st.session_state:
        st.session_state.historico = []

    retriever = obter_retriever(chroma_dir, collection_name, modelo_embedding)

    for i, mensagem in enumerate(st.session_state.historico):
        with st.chat_message(mensagem["papel"]):
            st.markdown(mensagem["conteudo"])

            if mensagem["papel"] == "assistant":
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button("👍 Correto", key=f"like-{i}"):
                        salvar_feedback(
                            message_id=mensagem["id"],
                            pergunta=mensagem["pergunta"],
                            resposta=mensagem["conteudo"],
                            valor_feedback=1,
                        )
                        st.success("Feedback positivo registrado.")
                with col2:
                    if st.button("👎 Impreciso", key=f"dislike-{i}"):
                        salvar_feedback(
                            message_id=mensagem["id"],
                            pergunta=mensagem["pergunta"],
                            resposta=mensagem["conteudo"],
                            valor_feedback=0,
                        )
                        st.warning("Feedback negativo registrado.")
                with col3:
                    st.caption("Sua avaliação ajuda o assistente a evoluir com base em dados reais.")

    if pergunta := st.chat_input("Digite sua pergunta sobre os documentos..."):
        st.session_state.historico.append({"papel": "user", "conteudo": pergunta})

        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            with st.spinner("Analisando contexto e gerando resposta..."):
                try:
                    resposta = gerar_resposta(pergunta, retriever, top_k, modelo_llm, ollama_url)
                except (ValueError, ErroOllama, RuntimeError) as erro:
                    resposta = f"Não foi possível gerar a resposta agora: {erro}"

            st.markdown(resposta)

            message_id = str(uuid4())
            st.session_state.historico.append(
                {
                    "id": message_id,
                    "papel": "assistant",
                    "conteudo": resposta,
                    "pergunta": pergunta,
                }
            )

            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("👍 Correto", key=f"like-novo-{message_id}"):
                    salvar_feedback(
                        message_id=message_id,
                        pergunta=pergunta,
                        resposta=resposta,
                        valor_feedback=1,
                    )
                    st.success("Feedback positivo registrado.")
            with col2:
                if st.button("👎 Impreciso", key=f"dislike-novo-{message_id}"):
                    salvar_feedback(
                        message_id=message_id,
                        pergunta=pergunta,
                        resposta=resposta,
                        valor_feedback=0,
                    )
                    st.warning("Feedback negativo registrado.")
            with col3:
                st.caption("Marque 👍 ou 👎 para alimentar o gráfico de aprendizado.")


if __name__ == "__main__":
    main()
