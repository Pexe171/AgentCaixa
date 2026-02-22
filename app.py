"""Fase 5: Interface Streamlit com chat e auditoria de lote."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from agent import ErroOllama, ErroOpenAI, responder_com_ollama, responder_com_openai
from query_rewriter import expandir_pergunta
from retriever import HybridRetriever

DB_PATH = Path("feedback.db")
RELATORIO_AVALIACAO_PATH = Path("relatorio_avaliacao.csv")
OPCOES_AVALIACAO_MANUAL = ["", "👍 Correto", "👎 Incorreto"]


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


def gerar_resposta(
    pergunta: str,
    retriever: HybridRetriever,
    top_k: int,
    modelo_llm: str,
    ollama_url: str,
    provedor: str,
) -> str:
    """Executa pipeline de recuperação + resposta final."""

    provedor_reescrita = "openai" if provedor == "openai" else "local"
    pergunta_tecnica = expandir_pergunta(pergunta, provedor=provedor_reescrita)
    documentos = retriever.buscar(pergunta_tecnica, top_k=top_k)
    if not documentos:
        return "[Informação não encontrada no documento]"

    documentos_dict = [asdict(item) for item in documentos]

    if provedor == "openai":
        return responder_com_openai(
            documentos=documentos_dict,
            pergunta=pergunta,
            modelo=modelo_llm,
        )

    return responder_com_ollama(
        documentos=documentos_dict,
        pergunta=pergunta,
        modelo=modelo_llm,
        base_url=ollama_url,
    )


def renderizar_sidebar() -> str:
    """Exibe opções globais na sidebar e retorna a tela selecionada."""

    modo = st.sidebar.radio("Navegação", options=["Chatbot", "Auditoria de Lote"], index=0)

    st.sidebar.header("Gráfico de Aprendizado")
    consolidado = carregar_aprendizado()

    if consolidado.empty:
        st.sidebar.info("Ainda não há feedbacks registrados.")
    else:
        taxa_media = consolidado["taxa_acerto"].mean()
        st.sidebar.metric("Taxa média de acerto", f"{taxa_media:.1f}%")
        st.sidebar.line_chart(consolidado.set_index("data")["taxa_acerto"])

    return modo


def renderizar_chatbot() -> None:
    """Renderiza a interface de chat e coleta de feedbacks."""

    st.title("💬 Assistente de Documentos")
    st.caption("Chat humanizado com feedback contínuo para evolução da qualidade das respostas.")

    with st.sidebar.expander("Configurações", expanded=True):
        chroma_dir = st.text_input("Diretório ChromaDB", value="./chroma_db")
        collection_name = st.text_input("Coleção", value="documentos")
        modelo_embedding = st.text_input("Modelo de embedding", value="nomic-embed-text")
        provedor_ui = st.radio(
            "Provedor de inferência",
            options=["Local (Ollama)", "Cloud (OpenAI)"],
            index=0,
        )
        if provedor_ui == "Cloud (OpenAI)":
            st.warning("Custo por token ativo")

        provedor = "openai" if provedor_ui == "Cloud (OpenAI)" else "local"
        modelo_padrao = "gpt-4o-mini" if provedor == "openai" else "llama3"
        modelo_llm = st.text_input("Modelo LLM", value=modelo_padrao)
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
                    resposta = gerar_resposta(pergunta, retriever, top_k, modelo_llm, ollama_url, provedor)
                except (ValueError, ErroOllama, ErroOpenAI, RuntimeError) as erro:
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


def renderizar_auditoria_lote() -> None:
    """Renderiza a tela de auditoria manual do relatório em lote."""

    st.title("🧪 Auditoria de Lote")
    st.caption("Revise respostas do relatório e registre a avaliação manual de cada linha.")

    if not RELATORIO_AVALIACAO_PATH.exists():
        st.warning(
            "O arquivo `relatorio_avaliacao.csv` ainda não foi encontrado. "
            "Rode `python avaliador_em_lote.py` para gerar o relatório e volte aqui."
        )
        return

    df = pd.read_csv(RELATORIO_AVALIACAO_PATH)

    if "Avaliação Manual" not in df.columns:
        df["Avaliação Manual"] = ""

    df["Avaliação Manual"] = df["Avaliação Manual"].fillna("")
    df["Avaliação Manual"] = df["Avaliação Manual"].where(
        df["Avaliação Manual"].isin(OPCOES_AVALIACAO_MANUAL),
        "",
    )

    colunas_bloqueadas = [coluna for coluna in df.columns if coluna != "Avaliação Manual"]

    df_editado = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        disabled=colunas_bloqueadas,
        column_config={
            "Avaliação Manual": st.column_config.SelectboxColumn(
                "Avaliação Manual",
                help="Classifique cada resposta do assistente.",
                options=OPCOES_AVALIACAO_MANUAL,
                required=False,
            )
        },
    )

    if st.button("Salvar Avaliações", type="primary"):
        df_editado.to_csv(RELATORIO_AVALIACAO_PATH, index=False)
        st.success("Avaliações salvas com sucesso em `relatorio_avaliacao.csv`.")


def main() -> None:
    """UI principal do chat e da auditoria em Streamlit."""

    st.set_page_config(page_title="Assistente de Documentos", page_icon="💬", layout="wide")

    inicializar_banco()
    modo = renderizar_sidebar()

    if modo == "Chatbot":
        renderizar_chatbot()
    else:
        renderizar_auditoria_lote()


if __name__ == "__main__":
    main()
