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

## 5) Infraestrutura local (recomendada)

O repositório inclui `docker-compose.infra.yml` com:

- `qdrant` (porta 6333)
- `redis` (porta 6379)

Subir infraestrutura:

```bash
docker compose -f docker-compose.infra.yml up -d
```

Parar infraestrutura:

```bash
docker compose -f docker-compose.infra.yml down
```

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
LLM_PROVIDER=mock
VECTOR_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_app_documents

EMBEDDING_CACHE_BACKEND=redis
EMBEDDING_CACHE_REDIS_URL=redis://localhost:6379/0

RESPONSE_CACHE_BACKEND=redis
RESPONSE_CACHE_REDIS_URL=redis://localhost:6379/1
```

---

## 8) Como executar

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

