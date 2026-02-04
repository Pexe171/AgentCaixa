# rag_app

Base do repositório para um app RAG em Python 3.11+.

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
