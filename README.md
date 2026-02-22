# Recomeço do Projeto

## Fase 1 — Ingestão de `.docx`

Foi adicionado o script `ingest_docx.py` para extrair texto com foco em precisão:

- Extrai parágrafos preservando a ordem do documento.
- Extrai tabelas linha por linha (cada linha vira um registro textual).
- Gera chunks conservadores de **até 400 caracteres** com **overlap de 150 caracteres**.
- Salva os chunks em JSON local temporário para auditoria.

## Fase 2 — Busca Híbrida (`retriever.py`)

Foi adicionado o script `retriever.py` para busca híbrida com foco em alta precisão:

- Usa **ChromaDB local** para armazenamento vetorial com persistência em disco.
- Usa embeddings via Ollama (padrão: `nomic-embed-text`).
- Implementa **BM25** para busca lexical (útil para siglas, códigos e números de leis/documentos).
- Faz fusão dos resultados com **RRF ponderado** (priorizando BM25 por padrão).



## Fase 3 — Agente de Resposta Final (`agent.py`)

Foi adicionado o script `agent.py` para montar a resposta final com LLM local:

- Conecta ao **Ollama local** (padrão: `llama3`).
- Usa **temperatura estritamente 0.0** para eliminar criatividade.
- Aplica um **System Prompt rigoroso** com regra explícita para ausência de informação:
  - `Você é um especialista analítico... [Informação não encontrada no documento]...`
- Recebe os documentos recuperados da Fase 2 e a pergunta do usuário para gerar a resposta final.

## Requisitos

- Python 3.10+
- Biblioteca `python-docx`
- Biblioteca `chromadb`
- Biblioteca `rank-bm25`
- Ollama ativo localmente (para geração de embeddings)

Instalação:

```bash
pip install python-docx chromadb rank-bm25 requests
```

## Como usar

### Fase 1 — gerar chunks

```bash
python ingest_docx.py caminho/arquivo.docx
```

Opcionalmente, você pode definir o caminho de saída do JSON:

```bash
python ingest_docx.py caminho/arquivo.docx --saida ./chunks_auditoria.json
```

### Fase 2 — indexar chunks no ChromaDB

```bash
python retriever.py --chunks-json ./chunks_auditoria.json --limpar
```

### Fase 2 — consultar (busca híbrida)

```bash
python retriever.py --pergunta "Qual é a vigência da norma X?" --top-k 8
```

### Fase 2 — indexar e consultar na mesma execução

```bash
python retriever.py \
  --chunks-json ./chunks_auditoria.json \
  --pergunta "Qual o prazo do documento Y?" \
  --top-k 8 \
  --limpar
```


### Fase 3 — recuperar contexto e gerar resposta final

```bash
python agent.py \
  --chunks-json ./chunks_auditoria.json \
  --pergunta "Qual é a vigência da norma X?" \
  --top-k 8 \
  --modelo-llm llama3
```

### Fase 3 — gerar resposta usando índice já existente

```bash
python agent.py --pergunta "Qual é o prazo do documento Y?"
```

## Saída

### `ingest_docx.py`

O script imprime:

- total de chunks gerados;
- caminho do arquivo JSON salvo.

Formato do JSON:

- `total_chunks`: quantidade total;
- `chunks`: lista com `id`, `tamanho` e `conteudo`.

### `retriever.py`

Quando recebe `--pergunta`, o script imprime:

- ID do chunk recuperado;
- score híbrido final;
- score BM25 normalizado;
- score vetorial aproximado;
- trecho do conteúdo recuperado.


### `agent.py`

Quando recebe `--pergunta`, o script:

- executa a recuperação de contexto com o retriever híbrido da Fase 2;
- envia contexto + pergunta para o Ollama com temperatura 0.0;
- imprime a resposta final do modelo;
- opcionalmente salva os documentos recuperados com `--salvar-contexto`.
