"""Interface de chat em Streamlit para o AgentCaixa."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000/v1/agent/chat")
TIMEOUT_SECONDS = float(os.getenv("AGENT_API_TIMEOUT_SECONDS", "45"))

st.set_page_config(page_title="Agente Caixa", page_icon="🤖", layout="wide")
st.title("🤖 Agente Caixa - Chat")
st.caption("Interface amigável para conversar com a API sem usar Swagger ou cURL.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configurações")
    llm_provider = st.selectbox(
        "Provedor de LLM",
        ["ollama", "padrao-backend", "openai", "mock"],
        index=0,
        help=(
            "Por padrão, a UI usa Ollama. "
            "Use padrao-backend para respeitar o servidor ou selecione openai/mock."
        ),
    )
    ollama_model = st.text_input(
        "Modelo Ollama (opcional)",
        value="llama3.2",
        help="Usado apenas quando o provedor selecionado for ollama.",
    )
    ollama_base_url = st.text_input(
        "URL do Ollama (opcional)",
        value="",
        help="Exemplo: http://localhost:11434",
    )
    st.divider()
    tone = st.selectbox(
        "Tom da resposta",
        ["profissional", "amigavel", "tecnico", "didatico"],
        index=0,
    )
    reasoning_depth = st.selectbox(
        "Profundidade",
        ["rapido", "padrao", "profundo"],
        index=1,
    )
    require_citations = st.toggle("Exigir citações", value=True)
    session_id = st.text_input("Session ID (opcional)", value="streamlit-session")
    st.divider()
    st.write(f"**API alvo:** `{API_URL}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Digite sua pergunta para o Agente Caixa")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "user_message": prompt,
        "session_id": session_id or None,
        "tone": tone,
        "reasoning_depth": reasoning_depth,
        "require_citations": require_citations,
    }

    if llm_provider != "padrao-backend":
        payload["llm_provider"] = llm_provider
    if llm_provider == "ollama" and ollama_model.strip():
        payload["ollama_model"] = ollama_model.strip()
    if llm_provider == "ollama" and ollama_base_url.strip():
        payload["ollama_base_url"] = ollama_base_url.strip()

    with st.chat_message("assistant"):
        with st.spinner("Consultando o Agente Caixa..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "Sem resposta.")
                citations = data.get("citations", [])
                diagnostics = data.get("diagnostics", {})

                content = answer
                if citations:
                    fontes = "\n".join(
                        (
                            f"- **{item.get('source', 'fonte')}** "
                            f"(score {item.get('score', 0):.2f})"
                        )
                        for item in citations[:5]
                    )
                    content += f"\n\n**Fontes principais**\n{fontes}"

                if diagnostics:
                    provider_used = str(diagnostics.get("provider_used", "")).lower()
                    if provider_used == "mock":
                        st.warning(
                            "A resposta foi gerada em MODO MOCK. "
                            "Configure OPENAI_API_KEY + OPENAI_MODEL "
                            "ou use override para Ollama para obter resposta real."
                        )
                    content += (
                        "\n\n<sub>"
                        f"provider={provider_used or diagnostics.get('provider_used')} | "
                        f"latência={diagnostics.get('latency_ms')}ms | "
                        f"trace_id={diagnostics.get('trace_id')}"
                        "</sub>"
                    )

                st.markdown(content, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )
            except requests.RequestException as exc:
                error = f"Falha ao consultar API: {exc}"
                st.error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error}
                )
