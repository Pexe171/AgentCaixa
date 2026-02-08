# rag_app

Aplicação Python 3.11+ para um agente **HAG (Hybrid Agentic Generation)** com foco em:

- chat inteligente com contexto;
- execução via OpenAI API ou **Ollama local**;
- varredura de pastas inteiras para análise de código, busca de riscos e apoio a debug.

## O que está implementado

- API FastAPI com:
  - `GET /health`
  - `POST /v1/agent/chat`
  - `POST /v1/agent/chat/stream` (SSE)
  - `POST /v1/agent/scan`
- Pipeline de chat com:
  - recuperação híbrida (lexical + vetorial) com reranking real dos 20 melhores candidatos e seleção final dos 5 mais relevantes;
  - estratégia **Small-to-Big Retrieval**: matching em frases menores com retorno do parágrafo completo para preservar contexto;
  - **Auto-Query Translation**: reescrita automática da pergunta para melhorar recall semântico na busca;
  - etapa explícita de planejamento (`Plano de Execução`) antes da resposta final;
  - montagem de prompt com tom/profundidade e memória de sessão;
  - geração por `mock`, `openai` ou `ollama`;
  - diagnósticos (latência, modelo, fallback);
  - endpoint de streaming em tempo real via SSE para reduzir percepção de latência.
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
- Memória semântica de longo prazo com sumarização periódica e recuperação vetorial por sessão.
- Orquestração multiagente com roteamento automático para especialistas (`analista_credito`, `especialista_juridico`, `atendimento_geral`).
- Guardrails de segurança com bloqueio de padrões maliciosos, detecção de PII/exfiltração e mudanças drásticas de comportamento com contexto de sessão.
- Observabilidade por resposta com `trace_id` e estimativa de custo por request.
- Resiliência HTTP com **exponential backoff + retries** para falhas transitórias e rate limits em OpenAI/Ollama.
- Cache de embeddings com backend em memória (padrão) e opção Redis para reduzir latência/custo em consultas repetidas.
- Recuperação vetorial avançada com suporte a `faiss` (com fallback local), além de provedores `qdrant`, `pgvector` e `weaviate`.
- Execução opcional de linters durante scan (`run_linters=true`).
- Suporte a **Tool Use** na integração OpenAI (`tools`), com execução de funções Python durante a resposta quando o modelo solicitar (ex.: `consultar_caixa(id_cliente)`).
- Dashboard administrativo em `/admin/dashboard` para acompanhar eventos de auditoria e notas do judge em tempo real.
- Parser DOCX com suporte a blocos de imagem via OCR (quando `pillow` + `pytesseract` estiverem instalados).

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
JUDGE_RESULTS_PATH=data/evals/judge_results.json
```

### Modo OpenAI

```bash
PROJECT_NAME=rag_app
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4.1-mini
OPENAI_API_KEY=<SUA_CHAVE_OPENAI>
OPENAI_TIMEOUT_SECONDS=20
```

### Tool Use (OpenAI)

O gateway OpenAI agora aceita o campo `tools` da API `/v1/responses` e executa o ciclo de tool calling:

1. modelo recebe as funções disponíveis;
2. se houver `function_call`, o backend executa a função Python;
3. o resultado volta como `function_call_output`;
4. o modelo gera a resposta final com base no retorno da ferramenta.

No fluxo de chat, a função `consultar_caixa` é disponibilizada para permitir consultas por `id_cliente`.


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

## Memória semântica de longo prazo

Além da janela curta de sessão, o agente agora cria resumos periódicos do diálogo e salva em SQLite com embedding vetorial para recall futuro.

```bash
SEMANTIC_MEMORY_BACKEND=sqlite
SEMANTIC_MEMORY_DB_PATH=data/memory/semantic_memory.db
SEMANTIC_MEMORY_RETRIEVE_TOP_K=3
SEMANTIC_MEMORY_SUMMARY_INTERVAL=4
```

- `SEMANTIC_MEMORY_SUMMARY_INTERVAL`: a cada N mensagens na sessão, um resumo factual é persistido.
- `SEMANTIC_MEMORY_RETRIEVE_TOP_K`: quantidade de memórias semânticas recuperadas e reinjetadas no prompt.

## Multi-Agent Systems (orquestração por domínio)

O pipeline agora possui um orquestrador leve por regras para escolher o especialista mais adequado antes de chamar o LLM:

- `analista_credito`: ativa quando a pergunta contém indícios de crédito, score, limite, financiamento e risco.
- `especialista_juridico`: ativa para termos contratuais, regulatórios, LGPD e compliance.
- `atendimento_geral`: fallback para demais temas.

O especialista roteado é exposto em `diagnostics.routed_specialist` e a justificativa em `diagnostics.routing_reason`.

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


## Refinamentos de RAG

### Small-to-Big Retrieval

A recuperação agora ranqueia trechos menores (frases) para aumentar precisão semântica, mas injeta no prompt o **bloco pai completo** (parágrafo/documento) para manter contexto rico na resposta final.

### Auto-Query Translation

Antes da busca lexical/vetorial, o agente executa uma etapa de tradução de consulta para reduzir ambiguidades. Exemplo:

- Entrada do usuário: `E a situação do Zé?`
- Consulta reescrita: `Qual é o estado atual do financiamento do cliente José da Silva?`

Essa tradução é usada apenas para retrieval/reranking; a pergunta original do usuário permanece no prompt final de resposta.

## Engenharia de produção e observabilidade

### Retries com exponential backoff

As chamadas HTTP para OpenAI e Ollama agora aplicam retentativas automáticas em erros transitórios (`429`, `500`, `502`, `503`, `504`, timeout e falhas de rede), com backoff exponencial.

### Cache de embeddings

O pipeline vetorial usa cache para reutilizar embeddings de textos/perguntas idênticas.

Configuração:

```bash
EMBEDDING_CACHE_BACKEND=memory   # opções: none | memory | redis
EMBEDDING_CACHE_REDIS_URL=redis://localhost:6379/0
EMBEDDING_CACHE_KEY_PREFIX=rag_app:embedding
EMBEDDING_CACHE_TTL_SECONDS=86400
```

Se `EMBEDDING_CACHE_BACKEND=redis` e o pacote `redis` não estiver disponível, o sistema usa fallback automático para cache em memória.


## Dashboard administrativo e integração de avaliação

A API expõe:

- `GET /admin/metrics`: retorna métricas agregadas de auditoria e avaliação do judge.
- `GET /admin/dashboard`: interface HTML simples para monitorização gráfica em tempo real.

O dashboard lê:

- `AUDIT_LOG_PATH` (trilha de auditoria do agente);
- `JUDGE_RESULTS_PATH` (saída JSON da avaliação do judge).

Para gerar o arquivo do judge:

```bash
python scripts/evaluate_agent_with_judge.py --output-path data/evals/judge_results.json
```

## OCR em documentos DOCX

O parser `rag_app/ingest/parser_docx.py` agora extrai texto de imagens incorporadas no DOCX (blocos `type="image"`).

Dependências opcionais:

```bash
pip install pillow pytesseract
```

> Observação: para OCR real também é necessário o binário do Tesseract instalado no sistema operacional.

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


### Chat com streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/v1/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Quero um plano completo para construir um agente de IA.",
    "tone": "didatico",
    "reasoning_depth": "profundo",
    "require_citations": true
  }'
```

O stream envia eventos `status`, `delta` e `done` no formato `text/event-stream`.

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


## Avaliação de respostas (LLM-as-a-Judge)

Foi adicionado o script `scripts/evaluate_agent_with_judge.py` para medir qualidade das respostas do agente em casos padrão.

### Modo heurístico local (sem API externa)

```bash
python scripts/evaluate_agent_with_judge.py --base-url http://localhost:8000
```

### Modo OpenAI (juiz externo)

```bash
python scripts/evaluate_agent_with_judge.py \
  --base-url http://localhost:8000 \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --openai-api-key "$OPENAI_API_KEY"
```

A saída é um JSON com nota por caso (1 a 10), justificativa e média consolidada.

## Testes

```bash
ruff check .
pytest -q
```
