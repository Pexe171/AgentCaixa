# Assistente RAG com Feedback de Aprendizado (Fases 1 a 4)

Este projeto implementa um pipeline completo de perguntas e respostas sobre documentos `.docx`, com busca híbrida, geração com LLM local e interface de chat no Streamlit com coleta de feedback.

## Visão geral da arquitetura

- **Fase 1 — Ingestão (`ingest_docx.py`)**
  - Extrai texto de parágrafos e tabelas de arquivos `.docx`.
  - Gera chunks conservadores (até 400 caracteres com overlap de 150).
  - Salva os chunks em JSON para auditoria e reuso.
- **Fase 2 — Recuperação híbrida (`retriever.py`)**
  - Indexa chunks no ChromaDB local (persistente em disco).
  - Realiza indexação em lotes de 50 chunks para reduzir timeout no embedding via Ollama.
  - Usa embedding via Ollama (`nomic-embed-text` por padrão).
  - Combina busca vetorial + BM25 com fusão RRF ponderada.
- **Fase 3 — Resposta final (`agent.py`)**
  - Recupera os melhores trechos via retriever híbrido.
  - Monta prompt estrito e chama Ollama (`llama3`, temperatura 0.0) com timeout padrão de 600 segundos.
  - Responde apenas com base no contexto recuperado.
- **Fase 4 — Interface (`app.py`)**
  - Chat humanizado em Streamlit.
  - Para cada resposta do bot: botões **👍 Correto** e **👎 Impreciso**.
  - Salva feedback em SQLite (`feedback.db`) com data, pergunta, resposta e feedback (1/0).
  - Exibe na sidebar o **Gráfico de Aprendizado** com taxa de acerto (%) ao longo do tempo.

---

## Pré-requisitos

- Python **3.10+**
- Ollama instalado e em execução local
- Pacote Python `ollama` (usado internamente pelo ChromaDB para embeddings via Ollama)
- Modelos Ollama disponíveis:
  - embeddings: `nomic-embed-text`
  - geração: `llama3`

Exemplo para preparar modelos no Ollama:

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

---

## Instalação

No diretório do projeto:

```bash
pip install python-docx chromadb rank-bm25 requests streamlit pandas ollama
```

---

## Execução do sistema completo

### 1) Gerar chunks do documento (Fase 1)

```bash
python ingest_docx.py caminho/arquivo.docx --saida ./chunks_auditoria.json
```

### 2) Indexar chunks no ChromaDB (Fase 2)

```bash
python retriever.py --chunks-json ./chunks_auditoria.json --limpar
```

> Dica: ajuste o tamanho de lote de indexação (padrão 50) com `--lote-indexacao` quando precisar otimizar estabilidade de embeddings.

### 3) (Opcional) Testar resposta via CLI (Fase 3)

```bash
python agent.py --pergunta "Qual é a vigência da norma X?"
```

Também é possível controlar o lote de indexação no fluxo da Fase 3 com `--lote-indexacao 50`.

### 4) Subir a interface web (Fase 4)

```bash
streamlit run app.py
```

Depois, abra no navegador o endereço mostrado pelo Streamlit (normalmente `http://localhost:8501`).

---

## Como usar o chat

1. Abra o app com `streamlit run app.py`.
2. Na **sidebar**, ajuste configurações como diretório do Chroma, coleção e modelos do Ollama.
3. O campo **Top-K de contexto** inicia em `4` por padrão (para reduzir latência); diminua para `3` se quiser ainda mais velocidade.
4. Digite sua pergunta no campo de chat.
5. Após cada resposta, clique em:
   - **👍 Correto** quando a resposta estiver adequada.
   - **👎 Impreciso** quando estiver incorreta ou incompleta.
6. A sidebar atualiza o **Gráfico de Aprendizado** com a taxa de acerto (%) por data.

---

## Banco de feedback (`feedback.db`)

A tabela `feedback` armazena:

- `data_hora` (timestamp da avaliação)
- `pergunta`
- `resposta`
- `feedback` (`1` para 👍 e `0` para 👎)
- `message_id` (identificador único da resposta para evitar duplicidade)

Esse banco é criado automaticamente na primeira execução do `app.py`.

---

## Estrutura dos arquivos principais

- `ingest_docx.py` — ingestão e chunking de `.docx`
- `retriever.py` — indexação e busca híbrida (vetorial + BM25)
- `agent.py` — geração final de resposta com Ollama
- `app.py` — interface Streamlit e coleta de feedback
- `feedback.db` — banco SQLite gerado em runtime

---

## Solução de problemas

- **Erro ao conectar no Ollama**
  - Verifique se o Ollama está ativo e acessível em `http://localhost:11434`.
- **Sem resultados na busca**
  - Reindexe com `--limpar` para reconstruir a base vetorial e BM25.
- **Gráfico de aprendizado vazio**
  - É esperado até existir pelo menos um feedback registrado.

---

## Próximas evoluções recomendadas

- Filtro por coleção/documento no chat.
- Dashboard com distribuição de feedback por tema.
- Exportação de feedback para CSV e rotinas de melhoria contínua do prompt.
