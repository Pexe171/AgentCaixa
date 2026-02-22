# Assistente RAG com Feedback de Aprendizado (Fases 1 a 6)

Este projeto implementa um pipeline completo de perguntas e respostas sobre documentos `.docx`, com busca híbrida, geração com LLM local, reescrita de perguntas para melhorar recuperação no banco vetorial, interface de chat no Streamlit com coleta de feedback e avaliador em lote para validação massiva.

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
  - Carrega o prompt de sistema a partir de ficheiros externos em `prompts/` (padrão: `especialista_habitacional.txt`).
  - Permite trocar o especialista via argumento `--prompt-sistema`.
  - Chama Ollama (`llama3`, temperatura 0.0) com timeout padrão de 600 segundos e responde apenas com base no contexto recuperado.
- **Fase 4 — Interface (`app.py`)**
  - Chat humanizado em Streamlit.
  - Para cada resposta do bot: botões **👍 Correto** e **👎 Impreciso**.
  - Salva feedback em SQLite (`feedback.db`) com data, pergunta, resposta e feedback (1/0).
  - Exibe na sidebar o **Gráfico de Aprendizado** com taxa de acerto (%) ao longo do tempo.
- **Fase 5 — Avaliador em lote (`avaliador_em_lote.py`)**
  - Lê `perguntas.txt` (uma pergunta por linha).
  - Recupera contexto com `HybridRetriever` usando **Top-K=4** (padrão).
  - Gera resposta para cada pergunta com `responder_com_ollama`.
  - Exporta `relatorio_avaliacao.csv` com colunas para auditoria e avaliação manual.
- **Fase 6 — Query Rewriting (`query_rewriter.py`)**
  - Reescreve perguntas coloquiais para uma versão técnica focada em normas habitacionais da Caixa, usando prompt externo `prompts/reescritor_tecnico.txt`.
  - Mantém cache em memória das perguntas reescritas para reduzir latência e chamadas repetidas ao Ollama.
  - Usa chamada rápida ao endpoint `http://localhost:11434/api/generate` com `requests` e timeout de 10s.
  - Em caso de erro/timeout, retorna a pergunta original como fallback seguro.

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

### 4) Subir a interface web (Fase 4 + Auditoria visual da Fase 5)

```bash
streamlit run app.py
```

Depois, abra no navegador o endereço mostrado pelo Streamlit (normalmente `http://localhost:8501`).

### 5) Rodar avaliador em lote (Fase 5)

Crie um arquivo `perguntas.txt` com uma pergunta por linha e execute:

```bash
python avaliador_em_lote.py
```

Saída padrão: `relatorio_avaliacao.csv`.


### 6) Auditar manualmente o relatório no Streamlit

1. Na barra lateral, selecione **Auditoria de Lote** em **Navegação**.
2. O app carregará `relatorio_avaliacao.csv` automaticamente.
3. Edite apenas a coluna **Avaliação Manual** usando as opções:
   - (vazio)
   - 👍 Correto
   - 👎 Incorreto
4. Clique em **Salvar Avaliações** para sobrescrever o CSV com suas marcações.

---


### Prompts externos especializados

A pasta `prompts/` centraliza os prompts de sistema e elimina textos fixos no código Python:

- `especialista_habitacional.txt` (padrão da Fase 3)
- `especialista_renda.txt`
- `reescritor_tecnico.txt` (Fase 6)

Para trocar o especialista na geração final (Fase 3), use:

```bash
python agent.py --pergunta "Minha pergunta" --prompt-sistema especialista_renda.txt
```

## Como usar o chat

1. Abra o app com `streamlit run app.py`.
2. Na **sidebar**, ajuste configurações como diretório do Chroma, coleção e modelos do Ollama.
3. O campo **Top-K de contexto** inicia em `4` por padrão (para reduzir latência); diminua para `3` se quiser ainda mais velocidade.
4. Digite sua pergunta no campo de chat. Antes da busca, o sistema aplica automaticamente Query Rewriting para transformar a pergunta em termos técnicos e melhorar a recuperação de contexto.
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
- `agent.py` — geração final de resposta com Ollama e carregamento de prompt externo
- `app.py` — interface Streamlit com dois modos: **Chatbot** e **Auditoria de Lote**, incluindo edição/salvamento da coluna `Avaliação Manual` no CSV
- `avaliador_em_lote.py` — execução em lote para validação e auditoria de respostas
- `query_rewriter.py` — reescrita técnica de perguntas com prompt externo (Query Rewriting)
- `prompts/` — prompts de sistema especializados por domínio
- `feedback.db` — banco SQLite gerado em runtime
- `perguntas.txt` — arquivo de entrada (uma pergunta por linha) para a Fase 5
- `relatorio_avaliacao.csv` — relatório gerado pela Fase 5

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

---

## Melhorias recentes de inteligência, performance e robustez

- **Mais rápido:** adicionado cache de Query Rewriting para perguntas repetidas, evitando chamadas redundantes ao Ollama.
- **Mais robusto:** validações extras no retriever híbrido (`k_rrf > 0` e ao menos um peso de busca maior que zero).
- **Mais resiliente:** alinhamento entre documentos, IDs e metadados no BM25 para evitar inconsistências quando houver itens inválidos.
- **Mais previsível para o usuário:** quando nenhum contexto é recuperado no chat, o sistema retorna imediatamente `[Informação não encontrada no documento]`.
