"""Avalia respostas do agente usando a abordagem LLM-as-a-Judge.

Uso rápido:
python scripts/evaluate_agent_with_judge.py \
  --base-url http://localhost:8000 \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --openai-api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class CasoAvaliacao:
    nome: str
    pergunta: str
    contexto_esperado: str


CASOS_PADRAO: list[CasoAvaliacao] = [
    CasoAvaliacao(
        nome="qualidade_arquitetura",
        pergunta=(
            "Quais melhorias priorizar para um agente de atendimento bancário "
            "com foco em segurança e observabilidade?"
        ),
        contexto_esperado=(
            "Esperado: sugerir guardrails, auditoria, monitoramento, métricas, "
            "testes e rollout incremental."
        ),
    ),
    CasoAvaliacao(
        nome="qualidade_rag",
        pergunta=(
            "Como aumentar precisão de respostas em um pipeline RAG híbrido "
            "com reranking?"
        ),
        contexto_esperado=(
            "Esperado: tratar recuperação lexical+vetorial, deduplicação, reranking, "
            "avaliação offline e monitoramento de qualidade."
        ),
    ),
    CasoAvaliacao(
        nome="qualidade_operacao",
        pergunta=(
            "Crie um checklist para colocar um agente de IA em produção "
            "com baixo risco operacional."
        ),
        contexto_esperado=(
            "Esperado: checklist de segurança, custos, fallback, "
            "SLO, alertas e plano de incidentes."
        ),
    ),
]


def chamar_agente(base_url: str, pergunta: str) -> dict[str, Any]:
    payload = {
        "user_message": pergunta,
        "tone": "profissional",
        "reasoning_depth": "profundo",
        "require_citations": True,
    }
    with httpx.Client(timeout=60.0) as client:
        resposta = client.post(f"{base_url.rstrip('/')}/v1/agent/chat", json=payload)
        resposta.raise_for_status()
        return resposta.json()


def _extrair_nota_judge(texto: str) -> tuple[float, str]:
    inicio = texto.find("{")
    fim = texto.rfind("}")
    if inicio == -1 or fim == -1 or fim <= inicio:
        raise ValueError("Judge retornou conteúdo sem JSON esperado.")

    payload = json.loads(texto[inicio : fim + 1])
    nota = float(payload["nota"])
    justificativa = str(payload.get("justificativa", ""))
    if nota < 1 or nota > 10:
        raise ValueError("Nota fora do intervalo esperado (1-10).")
    return nota, justificativa


def julgar_com_openai(
    judge_model: str,
    openai_api_key: str,
    pergunta: str,
    resposta_agente: str,
    contexto_esperado: str,
) -> tuple[float, str]:
    prompt_sistema = (
        "Você é um avaliador técnico extremamente rigoroso. "
        "Avalie a resposta do agente de 1 a 10 considerando: aderência ao contexto, "
        "correção, profundidade, objetividade e acionabilidade. "
        "Retorne SOMENTE JSON: {\"nota\": <numero>, \"justificativa\": \"texto\"}."
    )
    prompt_usuario = (
        f"Pergunta:\n{pergunta}\n\n"
        f"Contexto esperado:\n{contexto_esperado}\n\n"
        f"Resposta do agente:\n{resposta_agente}\n"
    )

    payload = {
        "model": judge_model,
        "input": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario},
        ],
    }

    with httpx.Client(timeout=60.0) as client:
        resposta = client.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resposta.raise_for_status()
        data = resposta.json()

    texto = str(data.get("output_text", "")).strip()
    if not texto:
        raise ValueError("Judge OpenAI retornou output_text vazio.")
    return _extrair_nota_judge(texto)


def julgar_heuristico(
    resposta_agente: str,
    contexto_esperado: str,
) -> tuple[float, str]:
    """Fallback local quando não há chave para juiz externo."""
    score = 5.0
    resposta_lower = resposta_agente.lower()
    for termo in ["segurança", "monitor", "fallback", "métrica", "teste", "rerank"]:
        if termo in resposta_lower:
            score += 0.7
    if len(resposta_agente) > 500:
        score += 0.5
    if "-" in resposta_agente or "1." in resposta_agente:
        score += 0.3
    score = min(10.0, max(1.0, round(score, 1)))
    return score, (
        "Avaliação heurística local aplicada por ausência de juiz externo. "
        f"Contexto esperado considerado: {contexto_esperado}"
    )


def _carregar_media_historica(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    media = data.get("media")
    if isinstance(media, (int, float)):
        return float(media)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação LLM-as-a-Judge do agente")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--judge-provider",
        choices=["openai", "heuristico"],
        default="heuristico",
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--output-path", default="data/evals/judge_results.json")
    parser.add_argument(
        "--baseline-path",
        default="data/evals/judge_baseline.json",
        help="Arquivo JSON com avaliação histórica para comparação.",
    )
    parser.add_argument(
        "--min-media",
        type=float,
        default=0.0,
        help="Se > 0, falha a execução quando a média atual ficar abaixo deste valor.",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.3,
        help=(
            "Queda máxima aceitável da média em relação ao baseline. "
            "Ex.: 0.3 permite cair até 0.3 ponto."
        ),
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Atualiza o baseline com o resultado atual após execução bem-sucedida.",
    )
    args = parser.parse_args()

    resultados: list[dict[str, Any]] = []
    for caso in CASOS_PADRAO:
        resposta = chamar_agente(base_url=args.base_url, pergunta=caso.pergunta)
        resposta_agente = str(resposta.get("answer", ""))

        if args.judge_provider == "openai":
            if not args.openai_api_key:
                raise ValueError(
                    "Para judge-provider=openai, informe --openai-api-key."
                )
            nota, justificativa = julgar_com_openai(
                judge_model=args.judge_model,
                openai_api_key=args.openai_api_key,
                pergunta=caso.pergunta,
                resposta_agente=resposta_agente,
                contexto_esperado=caso.contexto_esperado,
            )
        else:
            nota, justificativa = julgar_heuristico(
                resposta_agente=resposta_agente,
                contexto_esperado=caso.contexto_esperado,
            )

        resultados.append(
            {
                "caso": caso.nome,
                "nota": nota,
                "justificativa": justificativa,
                "resposta_agente": resposta_agente,
            }
        )

    media = round(statistics.mean(item["nota"] for item in resultados), 2)
    baseline_path = Path(args.baseline_path)
    media_baseline = _carregar_media_historica(baseline_path)
    regressao = (
        round(media_baseline - media, 2)
        if media_baseline is not None
        else None
    )

    saida = {
        "judge_provider": args.judge_provider,
        "judge_model": (
            args.judge_model
            if args.judge_provider == "openai"
            else "heuristico-local"
        ),
        "media": media,
        "media_baseline": media_baseline,
        "regressao": regressao,
        "resultados": resultados,
    }

    serialized = json.dumps(saida, ensure_ascii=False, indent=2)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)

    if args.min_media > 0 and media < args.min_media:
        raise SystemExit(
            f"Falha de qualidade: média {media} abaixo do mínimo {args.min_media}."
        )

    if regressao is not None and regressao > args.max_regression:
        raise SystemExit(
            "Falha de qualidade: regressão acima do limite. "
            f"Média atual={media}; baseline={media_baseline}; "
            f"limite={args.max_regression}."
        )

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
