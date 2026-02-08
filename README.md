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
  - recuperação de contexto local inicial;
  - montagem de prompt com tom/profundidade;
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

## Arquitetura resumida

### 1) Fluxo de chat (`/v1/agent/chat`)

1. Recebe `AgentChatRequest`.
2. Recupera snippets de contexto.
3. Monta prompts sistêmico + usuário.
4. Resolve provedor:
   - `openai` quando chave/modelo estiverem configurados;
   - `ollama` quando modelo local estiver configurado;
   - fallback para `mock` nos demais casos.
5. Retorna resposta com citações e diagnósticos.

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
    "max_files": 500
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

## Próximos passos para evolução “nível produção”

- Memória por sessão (`session_id`) com Redis/Postgres.
- Indexação vetorial real (pgvector/Qdrant/Weaviate).
- Execução opcional de linters por linguagem durante scan (flake8/eslint/golangci-lint etc.).
- Guardrails e políticas de segurança com trilha de auditoria.
- Observabilidade completa (OpenTelemetry, custo por request, tracing distribuído).
