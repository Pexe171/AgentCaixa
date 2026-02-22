# Recomeço do Projeto

## Fase 1 — Ingestão de `.docx`

Foi adicionado o script `ingest_docx.py` para extrair texto com foco em precisão:

- Extrai parágrafos preservando a ordem do documento.
- Extrai tabelas linha por linha (cada linha vira um registro textual).
- Gera chunks conservadores de **até 400 caracteres** com **overlap de 150 caracteres**.
- Salva os chunks em JSON local temporário para auditoria.

## Requisitos

- Python 3.10+
- Biblioteca `python-docx`

Instalação:

```bash
pip install python-docx
```

## Como usar

```bash
python ingest_docx.py caminho/arquivo.docx
```

Opcionalmente, você pode definir o caminho de saída do JSON:

```bash
python ingest_docx.py caminho/arquivo.docx --saida ./chunks_auditoria.json
```

## Saída

O script imprime:

- total de chunks gerados;
- caminho do arquivo JSON salvo.

Formato do JSON:

- `total_chunks`: quantidade total;
- `chunks`: lista com `id`, `tamanho` e `conteudo`.
