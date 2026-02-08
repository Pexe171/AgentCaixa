# AgentCaixa (rag_app)

Aplicação em **Python 3.11+** para execução de um agente conversacional com arquitetura RAG (retrieval augmented generation), memória de sessão, memória semântica e camadas de observabilidade.

> Este README foi reescrito do zero para deixar o projeto claro, prático e focado em uso local.

---

## 1) Objetivo do projeto

O AgentCaixa foi desenhado para:

- receber perguntas em linguagem natural;
- recuperar contexto relevante (busca lexical + vetorial);
- gerar resposta estruturada em português do Brasil;
- manter memória de interações da sessão;
- reduzir custo com cache de embeddings e cache de respostas repetidas;
- oferecer trilha de auditoria e métricas administrativas.

---

## 2) Pinecone é pago? Preciso usar?

### Resposta curta

- **Sim**, o Pinecone é um serviço gerenciado (cloud) e normalmente envolve plano pago conforme uso.
- **Não**, você **não precisa** usar Pinecone para rodar este projeto.

### Caminho recomendado para você (100% local)

Para rodar tudo localmente, use:

- **Qdrant** como banco vetorial local (via Docker);
- **Redis** para cache local (via Docker);
- `VECTOR_PROVIDER=qdrant`;
- sem chave de API de Pinecone.

Ou seja: dá para rodar localmente sem custo de Pinecone.

---

## 3) Arquitetura funcional

### 3.1 Pipeline de chat

1. Recebe requisição do usuário (`/v1/agent/chat`).
2. Aplica guardrails de segurança.
3. Reescreve a pergunta para busca (query translation).
4. Tenta cache de resposta completa (se habilitado).
5. Faz recuperação híbrida:
   - lexical;
   - vetorial (Qdrant/FAISS/fallback local, conforme configuração).
6. Reranqueia snippets.
7. Constrói plano de execução.
8. Gera resposta no provedor LLM configurado.
9. Persiste memória de sessão e memória semântica (quando aplicável).
10. Retorna resposta, citações e diagnósticos.

### 3.2 Camada vetorial

Provedores suportados:

- `none`: desativa busca vetorial.
- `faiss`: usa FAISS quando instalado; fallback local quando não disponível.
- `qdrant`: integração real com Qdrant.
- `pinecone`: integração com Pinecone (cloud, opcional).
- `pgvector` e `weaviate`: modo compatível com fallback local atual.

### 3.3 Cache

- **Cache de embeddings**: evita recomputar embeddings iguais.
- **Cache de respostas completas**: devolve resposta pronta para perguntas repetidas.

Backends disponíveis para ambos:

- `none`
- `memory`
- `redis`

---

## 4) Requisitos

- Python 3.11+
- pip
- Docker + Docker Compose (recomendado para infraestrutura local)

---

## 5) Subir tudo com **um único comando Docker**

Se você está em VPS com apenas um terminal, use o `docker-compose.yml` principal.

Ele sobe de uma vez:

- `api` (FastAPI + motor de raciocínio do agente)
- `frontend` (Streamlit)
- `qdrant` (vetorial)
- `redis` (cache)

### 5.1 Comando único (build + execução)

```bash
docker compose up --build -d
```

### 5.2 URLs de acesso

- Front-end: `http://localhost:8501`
- API: `http://localhost:8000`
- Healthcheck: `http://localhost:8000/health`

### 5.3 Parar tudo

```bash
docker compose down
```

### 5.4 Observação importante sobre o Qdrant no Docker Compose

O `docker-compose.yml` principal não usa mais healthcheck explícito para o serviço `qdrant`.

Motivo: em alguns ambientes, a imagem do Qdrant pode não trazer utilitários como `wget`, o que fazia o contêiner ser marcado como `unhealthy` mesmo com a API já disponível. Com isso, o `api` agora depende de `qdrant` em `service_started`, evitando falso negativo no boot.

> Se você quiser manter somente a infraestrutura (sem API/front), o arquivo legado `docker-compose.infra.yml` continua disponível.

---

## 6) Instalação da aplicação

### 6.1 Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### 6.2 Instalar dependências

Para uso local com Qdrant + Redis:

```bash
pip install -e .[dev,rag]
```

> O extra `rag` foi mantido focado no stack local (Redis + Qdrant).

Se você quiser testar Pinecone, instale manualmente:

```bash
pip install pinecone
```

---

## 7) Configuração de ambiente

Copie `.env.example` para `.env` e ajuste os valores:

```bash
cp .env.example .env
```

Configuração local recomendada:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
VECTOR_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_app_documents

EMBEDDING_CACHE_BACKEND=redis
EMBEDDING_CACHE_REDIS_URL=redis://localhost:6379/0

RESPONSE_CACHE_BACKEND=redis
RESPONSE_CACHE_REDIS_URL=redis://localhost:6379/1
```

---

### 7.1 Observabilidade com tracing (Langfuse ou LangSmith)

Para habilitar tracing detalhado do fluxo de orquestração:

```bash
OBSERVABILITY_ENABLED=true
OBSERVABILITY_PROVIDER=langsmith
LANGSMITH_API_KEY=<sua-chave>
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=agentcaixa
```

Alternativa com Langfuse:

```bash
OBSERVABILITY_ENABLED=true
OBSERVABILITY_PROVIDER=langfuse
LANGFUSE_PUBLIC_KEY=<public-key>
LANGFUSE_SECRET_KEY=<secret-key>
LANGFUSE_HOST=https://cloud.langfuse.com
```

Com isso, cada etapa do agente passa a ficar rastreável com spans e metadados para facilitar debug de falhas de raciocínio.

### 7.2 Parser DOCX para tabelas complexas

O parser padrão segue funcionando com `python-docx` + OCR, porém agora existe fallback automático para extração estruturada de tabelas com:

- **Docling** (prioridade 1);
- **Unstructured.io** (prioridade 2);
- serialização interna padrão (fallback final).

Para instalar extras de parser:

```bash
pip install -e .[parser]
```

## 8) Como executar sem Docker (modo desenvolvimento)

### 8.1 API

```bash
uvicorn rag_app.api.main:app --reload
```

### 8.2 Healthcheck

```bash
curl http://localhost:8000/health
```

### 8.3 Chat

```bash
curl -X POST http://localhost:8000/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Quero um plano para reduzir latência no meu agente.",
    "tone": "didatico",
    "reasoning_depth": "profundo",
    "require_citations": true
  }'
```

### 8.4 Override de provedor por requisição (ex.: Ollama/"Olami")

Quando você quiser forçar o uso do Ollama em uma chamada específica, envie `llm_provider` no body. O backend aceita alias comum (`olami`) e normaliza para `ollama`.

```bash
curl -X POST http://localhost:8000/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Me responda usando o modelo local.",
    "llm_provider": "olami",
    "ollama_model": "llama3.2",
    "ollama_base_url": "http://localhost:11434"
  }'
```

> Isso evita cair em resposta mock quando o provider padrão global não está em `ollama`.

---

## 9) Como funciona a parte vetorial na prática

### 9.1 Qdrant local (sem custo adicional de cloud)

Com `VECTOR_PROVIDER=qdrant`, o serviço:

- conecta no Qdrant local;
- sincroniza documentos da base interna;
- executa busca vetorial por similaridade;
- aplica fallback local se Qdrant indisponível.

### 9.2 Pinecone (opcional cloud)

Com `VECTOR_PROVIDER=pinecone`, o serviço tenta usar Pinecone.

Para isso você precisa:

- instalar SDK (`pip install pinecone`);
- informar `PINECONE_API_KEY`;
- ter índice criado no ambiente Pinecone.

Se não houver SDK/chave/disponibilidade, o serviço faz fallback local.

---

## 10) Cache de respostas: economia de tokens

Quando `RESPONSE_CACHE_BACKEND` está ativo, perguntas repetidas retornam do cache.

A chave de cache considera também o provedor/modelo configurado (ex.: `mock` vs `ollama`), evitando reaproveitar resposta de um backend em outro.

Efeito esperado:

- menor latência;
- menor uso de tokens no LLM;
- `diagnostics.provider_used` = `response-cache` em hit de cache.

---

## 11) Testes

Executar suíte:

```bash
pytest -q
```

---

## 12) Estrutura principal

- `rag_app/agent/service.py`: orquestração principal do chat.
- `rag_app/agent/vector_index.py`: retrieval vetorial e fallback.
- `rag_app/agent/embedding_cache.py`: cache de embeddings.
- `rag_app/agent/response_cache.py`: cache de respostas completas.
- `rag_app/config.py`: configuração central via variáveis de ambiente.
- `docker-compose.infra.yml`: infraestrutura local (Qdrant + Redis).

---

## 13) Resumo de decisão arquitetural

Para seu cenário (rodar no computador e evitar custo cloud):

- ✅ Use **Qdrant local** + **Redis local**.
- ✅ Mantenha Pinecone desabilitado.
- ✅ Ative cache de embeddings e respostas para reduzir custo de inferência.


---

## 11) Interface de Chat (Frontend) com Streamlit

Agora o projeto inclui uma interface moderna de chat para uso por usuários finais, sem precisar Swagger ou cURL.

Arquivo principal:

- `frontend/chat_app.py`

### 11.1 Instalação

```bash
pip install -e .[frontend]
```

### 11.2 Execução

Com a API FastAPI rodando na porta 8000:

```bash
streamlit run frontend/chat_app.py
```

Abrirá no navegador uma interface com:

- histórico de conversa;
- seleção de **provedor de LLM** (com Ollama como padrão da interface);
- campos opcionais de **modelo** e **URL do Ollama** para override por requisição;
- seleção de tom e profundidade de raciocínio;
- controle para exigir citações;
- exibição de diagnóstico (provider, latência e trace_id);
- alerta visual quando a resposta vier em **MODO MOCK**.

Você pode trocar o endpoint da API via variável:

```bash
export AGENT_API_URL=http://localhost:8000/v1/agent/chat
```


### 11.3 Por que aparece `[MODO MOCK]` no chat?

Quando OpenAI/Ollama não estão corretamente configurados ou indisponíveis, o serviço faz fallback para o provedor mock.

No setup padrão deste repositório, o `docker-compose.yml` sobe com `LLM_PROVIDER=ollama` e `OLLAMA_MODEL=llama3.2`.

Como resolver:

1. Configure variáveis de ambiente válidas para OpenAI (`OPENAI_API_KEY` + `OPENAI_MODEL`) e use `LLM_PROVIDER=openai`; **ou**
2. No próprio front-end, selecione `ollama` no campo **Provedor de LLM** e informe `Modelo Ollama` (e opcionalmente `URL do Ollama`).

Assim, a requisição envia `llm_provider` no payload e evita fallback involuntário para mock.

---

## 12) Integração com WhatsApp via Evolution API

Foi adicionada integração de canal com WhatsApp para analistas consultarem o AgentCaixa diretamente no celular.

### 12.1 Endpoint de webhook

```http
POST /v1/channels/whatsapp/evolution/webhook
```

O endpoint:

1. valida segredo de webhook (opcional);
2. interpreta evento `messages.upsert` da Evolution API;
3. chama o fluxo `/v1/agent/chat` internamente;
4. envia a resposta de volta para o WhatsApp.

### 12.2 Variáveis de ambiente

```bash
WHATSAPP_CHANNEL_ENABLED=true
WHATSAPP_PROVIDER=evolution
WHATSAPP_EVOLUTION_BASE_URL=http://localhost:8080
WHATSAPP_EVOLUTION_INSTANCE=agentecaixa
WHATSAPP_EVOLUTION_API_KEY=seu_token
WHATSAPP_WEBHOOK_SECRET=segredo_compartilhado
```

### 12.3 Exemplo de registro de webhook na Evolution

No painel/instância da Evolution API, configure o webhook para apontar para:

```text
https://SEU_HOST/v1/channels/whatsapp/evolution/webhook
```

Inclua o header:

```text
x-webhook-secret: segredo_compartilhado
```

> Dica: em ambiente de desenvolvimento local, use túnel (ex.: ngrok/cloudflared) para expor o FastAPI.
## 14) Observabilidade com Langfuse (trace por etapa)

Agora o agente possui integração opcional com **Langfuse** para rastrear o pipeline completo (traces e spans), incluindo etapas como:

- tradução de query para retrieval;
- consulta de cache de respostas;
- recuperação híbrida (lexical + vetorial + reranking);
- planejamento de execução;
- geração da resposta final.

### 14.1 Dependência

Instale o extra de observabilidade:

```bash
pip install -e .[observability]
```

### 14.2 Configuração

No `.env`:

```bash
OBSERVABILITY_ENABLED=true
OBSERVABILITY_PROVIDER=langfuse
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=seu_public_key
LANGFUSE_SECRET_KEY=seu_secret_key
```

Se a integração não estiver configurada, o sistema usa fallback seguro (no-op), sem quebrar o fluxo do agente.

---

## 15) Auto-avaliação contínua em cada deploy

Foi preparada automação para rodar `scripts/evaluate_agent_with_judge.py` em pipeline CI/CD, com proteção contra regressão de qualidade.

### 15.1 Como a regressão é detectada

O script agora suporta:

- `--baseline-path`: referência histórica de qualidade;
- `--min-media`: nota média mínima obrigatória;
- `--max-regression`: queda máxima aceitável versus baseline;
- `--update-baseline`: atualiza baseline após execução bem-sucedida.

Exemplo local:

```bash
python scripts/evaluate_agent_with_judge.py \
  --base-url http://localhost:8000 \
  --judge-provider heuristico \
  --output-path data/evals/judge_results.json \
  --baseline-path data/evals/judge_baseline.json \
  --min-media 6.0 \
  --max-regression 0.5
```

### 15.2 Pipeline recomendado

No GitHub Actions (`.github/workflows/evaluate-agent.yml`), a API sobe localmente e executa a avaliação.

- Se `OPENAI_API_KEY` estiver disponível, roda com judge OpenAI.
- Sem chave, roda com judge heurístico local.
- O job falha automaticamente quando a qualidade regrede além do limite definido.

## 16) Como fazer o AgentCaixa "aprender" com DOCX (e ajustar parâmetros)

Se a sua ideia é ensinar o agente com conteúdo interno (procedimentos, normas, playbooks), o fluxo recomendado é:

1. converter/normalizar documentos;
2. transformar o conteúdo em blocos utilizáveis;
3. ativar busca vetorial + memória semântica;
4. ajustar parâmetros de recuperação para melhorar relevância.

### 16.1 Ingestão de DOCX (CLI)

O projeto já traz comando de ingestão via CLI:

```bash
python -m rag_app.cli ingest --input ./docs/meu_procedimento.docx
```

Esse comando:

- converte o arquivo para DOCX quando necessário;
- extrai blocos de conteúdo;
- gera um `data/processed/blocks.jsonl` com os trechos processados.

Você pode repetir para novos documentos e versionar seu diretório `docs/` para manter histórico de conhecimento.

### 16.2 Formatos além de DOCX

Para arquivos mais complexos (DOCX com tabelas e layouts), instale os extras de parser:

```bash
pip install -e .[parser]
```

Com isso, o pipeline tenta extração estruturada com fallback automático entre Docling/Unstructured e parser interno.

### 16.3 Ativando vetores de aprendizado (RAG vetorial)

No `.env`, habilite um provider vetorial (recomendação local: Qdrant):

```bash
VECTOR_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_app_documents

EMBEDDING_CACHE_BACKEND=redis
EMBEDDING_CACHE_REDIS_URL=redis://localhost:6379/0
```

E suba a infra:

```bash
docker compose -f docker-compose.infra.yml up -d
```

### 16.4 Parâmetros para melhorar qualidade de resposta

Você pode calibrar estes parâmetros no `.env` para evoluir o comportamento do agente:

- `RETRIEVE_TOP_K_DEFAULT`: quantidade de trechos recuperados por busca.
- `HYBRID_ALPHA`: peso entre estratégias de recuperação.
- `STRICT_MIN_SCORE`: corte mínimo de relevância.
- `MAX_SNIPPET_CHARS`: tamanho máximo de cada trecho no contexto.
- `SEMANTIC_MEMORY_RETRIEVE_TOP_K`: quantas memórias semânticas antigas recuperar.
- `SEMANTIC_MEMORY_SUMMARY_INTERVAL`: frequência de sumarização da memória.

Exemplo de ajuste inicial:

```bash
RETRIEVE_TOP_K_DEFAULT=8
HYBRID_ALPHA=0.70
STRICT_MIN_SCORE=0.25
MAX_SNIPPET_CHARS=280
SEMANTIC_MEMORY_RETRIEVE_TOP_K=4
SEMANTIC_MEMORY_SUMMARY_INTERVAL=3
```

### 16.5 Ciclo prático de melhoria contínua

1. Ingerir documentos reais da operação (`python -m rag_app.cli ingest ...`).
2. Rodar perguntas de validação no `/v1/agent/chat`.
3. Medir com `scripts/evaluate_agent_with_judge.py`.
4. Ajustar parâmetros (`TOP_K`, `HYBRID_ALPHA`, score mínimo).
5. Repetir ciclo até estabilizar qualidade + latência.


## 17) Solução de problemas (troubleshooting)

### Erro: `ImportError: cannot import name 'scan_folder'`

Se ao subir a API aparecer erro de importação em `rag_app.agent.scanner`, confirme que o arquivo contém a função `scan_folder(...)` e que você está executando a versão mais recente do projeto.

Validação rápida:

```bash
python -c "from rag_app.agent.scanner import scan_folder; print('ok')"
```

### Erro de dependência opcional (`ModuleNotFoundError: langchain_community`)

Alguns módulos vetoriais usam extras opcionais. Instale o pacote com extras para ambiente de desenvolvimento:

```bash
pip install -e .[dev,rag]
```


---

## 8) Solução de erro comum ao subir a API

Se ao iniciar com `uvicorn rag_app.api.main:app` aparecer erro de importação envolvendo `rag_app.agent.vector_index` (por exemplo `_cosine_similarity`), o projeto já inclui fallback local determinístico para embeddings e similaridade.

Checklist rápido:

1. Atualize o código para a versão mais recente da branch.
2. Garanta que o ambiente virtual ativo corresponde ao diretório atual.
3. Rode os testes de recuperação vetorial:

```bash
pytest -q tests/test_vector_index.py
```

Esse fluxo valida:

- providers vetoriais em modo fallback local;
- estratégia de ranqueamento híbrida (semântica + lexical);
- cache de embeddings em memória;
- compatibilidade de importação do serviço principal da API.
