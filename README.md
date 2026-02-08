# rag_app

Aplicação Python 3.11+ para um agente **HAG (Hybrid Agentic Generation)** com foco em:

- chat inteligente com contexto;
- execução via OpenAI API ou **Ollama local**;
- varredura de pastas inteiras para análise de código, busca de riscos e apoio a debug.

## O que está implementado

- API FastAPI com:
  - `GET /health`
  - `POST /v1/agent/chat`
  - `POST /v1/agent/scan`
- Pipeline de chat com:
  - recuperação híbrida (lexical + vetorial) com reranking real dos 20 melhores candidatos e seleção final dos 5 mais relevantes;
  - etapa explícita de planejamento (`Plano de Execução`) antes da resposta final;
  - montagem de prompt com tom/profundidade e memória de sessão;
  - geração por `mock`, `openai` ou `ollama`;
  - diagnósticos (latência, modelo, fallback).
- Scanner multi-linguagem de pasta para identificar problemas comuns:
  - possível segredo hardcoded;
  - logs de debug residuais;
  - tratamento de erro frágil;
  - código comentado suspeito;
  - TODO/FIXME pendente.
- CLI com comandos:
  - `ingest` (documentos)
  - `scan` (varredura local de código)
- Testes automatizados para API, serviço, scanner e configuração.
- Memória por sessão (`session_id`) para continuidade de contexto em múltiplas mensagens.
- Persistência de memória em banco SQLite (configurável) para manter histórico entre reinicializações.
- Guardrails de segurança com bloqueio de padrões maliciosos e trilha de auditoria.
- Observabilidade por resposta com `trace_id` e estimativa de custo por request.
- Recuperação vetorial avançada com suporte a `faiss` (com fallback local), além de provedores `qdrant`, `pgvector` e `weaviate`.
- Execução opcional de linters durante scan (`run_linters=true`).

## Arquitetura resumida

### 1) Fluxo de chat (`/v1/agent/chat`)

1. Recebe `AgentChatRequest`.
2. Recupera até 20 snippets lexicais + 20 vetoriais.
3. Deduplica resultados e aplica reranking para escolher os 5 melhores contextos.
4. Gera um Plano de Execução com o LLM antes da resposta final.
5. Monta prompts sistêmico + usuário com plano + contexto final.
6. Resolve provedor:
   - `openai` quando chave/modelo estiverem configurados;
   - `ollama` quando modelo local estiver configurado;
   - fallback para `mock` nos demais casos.
7. Retorna resposta com citações e diagnósticos.

### 2) Fluxo de análise de pasta (`/v1/agent/scan`)

1. Recebe caminho da pasta (`folder_path`).
2. Percorre arquivos suportados por extensão.
3. Executa regras simples de análise linha a linha.
4. Consolida relatório com issues + resumo executivo.

## Configuração (`.env`)

### Modo mock (desenvolvimento rápido)

```bash
PROJECT_NAME=rag_app
LLM_PROVIDER=mock
RETRIEVE_TOP_K_DEFAULT=6
VECTOR_PROVIDER=none
SESSION_STORE_BACKEND=memory
SESSION_DB_PATH=data/memory/session_memory.db
ENABLE_LINTER_SCAN=false
AUDIT_LOG_PATH=data/audit/agent_audit.log
COST_PER_1K_TOKENS_USD=0.002
```

### Modo OpenAI

```bash
PROJECT_NAME=rag_app
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4.1-mini
OPENAI_API_KEY=<SUA_CHAVE_OPENAI>
OPENAI_TIMEOUT_SECONDS=20
```

### Modo Ollama (local)

```bash
PROJECT_NAME=rag_app
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT_SECONDS=30
```


## Persistência de memória de conversa

Por padrão, a memória de sessão roda em `memory` (somente processo atual).
Para persistência real entre reinícios da API, ative SQLite:

```bash
SESSION_STORE_BACKEND=sqlite
SESSION_DB_PATH=data/memory/session_memory.db
```

## RAG vetorial avançado

O projeto agora possui recuperação vetorial com embeddings densos determinísticos e ranqueamento por similaridade de cosseno:

- `VECTOR_PROVIDER=faiss`: usa pipeline compatível com FAISS; se o pacote não estiver instalado, aplica fallback local automático com a mesma estratégia vetorial.
- `VECTOR_PROVIDER=qdrant`: usa pipeline vetorial local compatível para facilitar migração para Qdrant.
- `VECTOR_PROVIDER=none`: desativa camada vetorial.

Exemplo:

```bash
VECTOR_PROVIDER=faiss
RETRIEVE_TOP_K_DEFAULT=6
```

## Execução local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn rag_app.api.main:app --reload
```

## Exemplos de uso da API

### Health

```bash
curl http://localhost:8000/health
```

### Chat

```bash
curl -X POST http://localhost:8000/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Quero um plano completo para construir um agente de IA.",
    "tone": "didatico",
    "reasoning_depth": "profundo",
    "require_citations": true
  }'
```

### Scan de pasta para debug

```bash
curl -X POST http://localhost:8000/v1/agent/scan \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/workspace/AgentCaixa",
    "include_hidden": false,
    "max_files": 500,
    "run_linters": true
  }'
```

## Uso via CLI

### Varredura local

```bash
python -m rag_app.cli scan --folder /workspace/AgentCaixa --max-files 500
```

## Testes

```bash
ruff check .
pytest -q
```
