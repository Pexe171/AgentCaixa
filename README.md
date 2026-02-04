# rag_app

Base do repositório para um app RAG em Python 3.11+.

## Configuração

Crie um arquivo `.env` na raiz para definir variáveis de ambiente. Exemplo:

```bash
PROJECT_NAME=rag_app
LOG_LEVEL=INFO
RETRIEVE_TOP_K_DEFAULT=6
```

As configurações são carregadas em `rag_app.config.load_settings()` e podem ser sobrescritas
via variáveis de ambiente.

## Como executar localmente

Crie um ambiente virtual e instale as dependências (a definir).

Execute a API:

```bash
uvicorn rag_app.api.main:app --reload
```

Teste o health check:

```bash
curl http://localhost:8000/health
```
