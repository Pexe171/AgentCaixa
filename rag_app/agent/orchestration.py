"""Regras de orquestração para roteamento entre especialistas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialistDecision:
    specialist: str
    reason: str


def route_to_specialist(user_message: str) -> SpecialistDecision:
    message = user_message.lower()

    credit_keywords = {
        "credito",
        "crédito",
        "score",
        "limite",
        "financiamento",
        "renda",
        "inadimplencia",
        "inadimplência",
    }
    legal_keywords = {
        "juridico",
        "jurídico",
        "contrato",
        "compliance",
        "regulatorio",
        "regulatório",
        "lgpd",
        "lei",
        "resolução",
    }

    if any(keyword in message for keyword in credit_keywords):
        return SpecialistDecision(
            specialist="analista_credito",
            reason="Detectados termos de análise de crédito e risco financeiro.",
        )

    if any(keyword in message for keyword in legal_keywords):
        return SpecialistDecision(
            specialist="especialista_juridico",
            reason="Detectados termos jurídicos, regulatórios ou contratuais.",
        )

    return SpecialistDecision(
        specialist="atendimento_geral",
        reason="Sem gatilhos de domínio específico; manter fluxo geral.",
    )


def specialist_instruction(specialist: str) -> str:
    if specialist == "analista_credito":
        return (
            "Atue como Analista de Crédito. Explique critérios de risco, "
            "documentação necessária, impactos de renda/score e próximos passos."
        )
    if specialist == "especialista_juridico":
        return (
            "Atue como Especialista Jurídico. Traga alertas regulatórios, "
            "riscos legais e recomendações de conformidade de forma objetiva."
        )
    return (
        "Atue como Especialista de Atendimento Geral. Organize a resposta "
        "de forma clara, prática e orientada a ação."
    )
