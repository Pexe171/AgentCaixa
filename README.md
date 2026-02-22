# Assistente RAG com Feedback de Aprendizado (Fases 1 a 6)

Este projeto implementa um pipeline completo de perguntas e respostas sobre documentos `.docx`, com busca híbrida, geração Multi-LLM (Ollama, OpenAI e Gemini), reescrita de perguntas para melhorar recuperação no banco vetorial, interface de chat no Streamlit com coleta de feedback e avaliador em lote para validação massiva.

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
  - Suporta arquitetura híbrida com `Ollama`, `OpenAI` e `Google Gemini`, sempre com temperatura 0.0 e prompts externos em `prompts/`.
- **Fase 4 — Interface (`app.py`)**
  - Chat humanizado em Streamlit.
  - Para cada resposta do bot: botões **👍 Correto** e **👎 Impreciso**.
  - Salva feedback em SQLite (`feedback.db`) com data, pergunta, resposta e feedback (1/0).
  - Exibe na sidebar o **Gráfico de Aprendizado** com taxa de acerto (%) ao longo do tempo.
- **Fase 5 — Avaliador em lote (`avaliador_em_lote.py`)**
  - Lê `perguntas.txt` (uma pergunta por linha).
  - Recupera contexto com `HybridRetriever` usando **Top-K=4** (padrão).
  - Gera resposta para cada pergunta com roteamento automático por provedor (`gerar_resposta_hibrida`).
  - Para OpenAI e Gemini, usa paralelismo por threads para acelerar a geração do Relatório de Ouro.
  - Exporta CSV com colunas para auditoria e avaliação manual.
- **Fase 6 — Query Rewriting (`query_rewriter.py`)**
  - Reescreve perguntas coloquiais para uma versão técnica focada em normas habitacionais da Caixa, usando prompt externo `prompts/reescritor_tecnico.txt`.
  - Mantém cache em memória das perguntas reescritas para reduzir latência e chamadas repetidas ao Ollama.
  - Usa chamada rápida ao endpoint `http://localhost:11434/api/generate` com `requests` e timeout de 10s.
  - Em caso de erro/timeout, retorna a pergunta original como fallback seguro.

---


## 🛠️ Configuração de Motores de IA

O AgentCaixa agora suporta três provedores de geração: **Ollama (local)**, **OpenAI** e **Google Gemini**.

### 1) Obter chave da OpenAI

1. Acesse o painel da OpenAI.
2. Gere uma API key em **API Keys**.
3. Copie a chave para uso no `.env`.

### 2) Obter chave do Google AI Studio (Gemini)

1. Acesse o Google AI Studio.
2. Crie uma API key para a API Gemini.
3. Copie a chave para uso no `.env`.

### 3) Configurar o arquivo `.env`

Na raiz do projeto, crie (ou edite) o arquivo `.env` com as duas chaves:

```bash
OPENAI_API_KEY=sua_chave_openai_aqui
GOOGLE_API_KEY=sua_chave_google_ai_studio_aqui
```

> O carregamento do `.env` é automático nas integrações cloud do sistema (OpenAI/Gemini).

### 4) Executar avaliador em lote por provedor

```bash
# Ollama (local)
python avaliador_em_lote.py --provedor local --modelo-llm llama3

# OpenAI (Relatório de Ouro com threads)
python avaliador_em_lote.py --provedor openai --threads 50 --modelo-llm gpt-4o-mini

# Gemini (Relatório de Ouro com threads)
python avaliador_em_lote.py --provedor gemini --threads 50 --modelo-llm gemini-1.5-flash
```

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
pip install python-docx chromadb rank-bm25 requests streamlit pandas ollama openai google-generativeai python-dotenv
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
python avaliador_em_lote.py --provedor local
```

Saída padrão local: `relatorio_avaliacao.csv`.

#### Modo Turbo (OpenAI com paralelismo)

```bash
python avaliador_em_lote.py --provedor openai --threads 50 --modelo-llm gpt-4o-mini
```

Saída padrão OpenAI: `relatorio_ouro_openai.csv`.


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
2. Na **sidebar**, ajuste configurações como diretório do Chroma, coleção e modelos.
3. Selecione o **Provedor de IA** em `Ollama`, `OpenAI` ou `Gemini`.
4. O app atualiza dinamicamente a lista de **Modelo LLM** conforme o provedor escolhido.
5. Quando `OpenAI` ou `Gemini` estiver ativo, o app exibirá o aviso **Custo por token ativo**.
6. O campo **Top-K de contexto** inicia em `4` por padrão (para reduzir latência); diminua para `3` se quiser ainda mais velocidade.
7. Digite sua pergunta no campo de chat. Antes da busca, o sistema aplica automaticamente Query Rewriting para melhorar a recuperação de contexto.
8. Após cada resposta, clique em:
   - **👍 Correto** quando a resposta estiver adequada.
   - **👎 Impreciso** quando estiver incorreta ou incompleta.
9. A sidebar atualiza o **Gráfico de Aprendizado** com a taxa de acerto (%) por data.

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
- `agent.py` — geração final com roteamento híbrido entre Ollama, OpenAI e Gemini, com carregamento de prompt externo
- `app.py` — interface Streamlit com dois modos: **Chatbot** e **Auditoria de Lote**, incluindo edição/salvamento da coluna `Avaliação Manual` no CSV
- `avaliador_em_lote.py` — execução em lote para validação e auditoria de respostas (inclui modo concorrente para OpenAI/Gemini)
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
