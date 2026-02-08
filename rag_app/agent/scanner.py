"""Varredura multi-linguagem para encontrar erros comuns e orientar debug."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rag_app.agent.schemas import AgentScanResponse, ScanIssue

SUPPORTED_SUFFIXES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".sh": "shell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
}

SECRET_HINTS = (
    "api_key",
    "secret",
    "token",
    "password",
    "-----BEGIN",
)


def _iter_candidate_files(
    root: Path,
    include_hidden: bool,
    max_files: int,
) -> list[Path]:
    candidates: list[Path] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if not include_hidden and any(part.startswith(".") for part in file_path.parts):
            continue
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        candidates.append(file_path)
        if len(candidates) >= max_files:
            break
    return candidates


def _evaluate_line(file_path: Path, line: str, line_number: int) -> list[ScanIssue]:
    lower_line = line.lower()
    issues: list[ScanIssue] = []

    if any(hint in lower_line for hint in SECRET_HINTS) and "=" in line:
        issues.append(
            ScanIssue(
                file_path=str(file_path),
                line_number=line_number,
                severity="alta",
                category="possivel-segredo",
                message="Possível segredo hardcoded no código.",
                suggestion=(
                    "Mover segredo para variável de ambiente "
                    "e rotacionar credencial."
                ),
            )
        )

    if "print(" in lower_line or "console.log(" in lower_line:
        issues.append(
            ScanIssue(
                file_path=str(file_path),
                line_number=line_number,
                severity="media",
                category="debug-residual",
                message="Log de depuração residual encontrado.",
                suggestion=(
                    "Substituir por logger com níveis "
                    "e remover logs temporários."
                ),
            )
        )

    if "todo" in lower_line or "fixme" in lower_line:
        issues.append(
            ScanIssue(
                file_path=str(file_path),
                line_number=line_number,
                severity="baixa",
                category="boas-praticas",
                message="Marcador pendente identificado (TODO/FIXME).",
                suggestion="Registrar em backlog e tratar com critério de prioridade.",
            )
        )

    if lower_line.strip().startswith("#") and "=" in line and len(line) > 50:
        issues.append(
            ScanIssue(
                file_path=str(file_path),
                line_number=line_number,
                severity="baixa",
                category="codigo-comentado",
                message="Possível bloco de código comentado.",
                suggestion=(
                    "Remover código morto e preservar "
                    "histórico no versionamento."
                ),
            )
        )

    if "except:" in lower_line or ("catch (" in lower_line and "{}" in lower_line):
        issues.append(
            ScanIssue(
                file_path=str(file_path),
                line_number=line_number,
                severity="alta",
                category="tratamento-de-erros",
                message="Tratamento de erro muito genérico ou vazio.",
                suggestion=(
                    "Capturar exceções específicas e registrar "
                    "contexto para facilitar debug."
                ),
            )
        )

    return issues


def _run_optional_linters(root: Path, languages: set[str]) -> list[str]:
    linter_commands: dict[str, list[str]] = {
        "python": ["flake8", str(root)],
        "javascript": ["eslint", str(root)],
        "typescript": ["eslint", str(root)],
        "go": ["golangci-lint", "run", str(root)],
    }

    findings: list[str] = []
    for language in sorted(languages):
        command = linter_commands.get(language)
        if not command:
            continue
        if not shutil.which(command[0]):
            findings.append(
                f"Linter para {language} não disponível no ambiente ({command[0]})."
            )
            continue

        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            findings.append(f"{command[0]} executado sem erros para {language}.")
            continue

        output = (completed.stdout or completed.stderr).strip().splitlines()
        sample = output[0] if output else "linter retornou erro sem detalhes"
        findings.append(f"{command[0]} reportou achados em {language}: {sample}")

    return findings


def scan_folder(
    folder_path: str,
    include_hidden: bool = False,
    max_files: int = 400,
    run_linters: bool = False,
) -> AgentScanResponse:
    """Executa varredura da pasta procurando problemas comuns."""

    root = Path(folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError("folder_path deve existir e ser um diretório.")

    files = _iter_candidate_files(
        root=root,
        include_hidden=include_hidden,
        max_files=max_files,
    )

    issues: list[ScanIssue] = []
    languages: set[str] = set()

    for file_path in files:
        languages.add(SUPPORTED_SUFFIXES[file_path.suffix.lower()])
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for index, line in enumerate(content.splitlines(), start=1):
            line_issues = _evaluate_line(
                file_path=file_path,
                line=line,
                line_number=index,
            )
            issues.extend(line_issues)

    linter_findings = (
        _run_optional_linters(root=root, languages=languages) if run_linters else []
    )

    summary = (
        "Varredura concluída com sucesso. "
        f"Arquivos analisados: {len(files)}. "
        f"Linguagens detectadas: {', '.join(sorted(languages)) or 'nenhuma'}. "
        f"Achados totais: {len(issues)}. "
        f"Linters executados: {'sim' if run_linters else 'não'}."
    )

    return AgentScanResponse(
        folder_path=str(root),
        files_scanned=len(files),
        languages_detected=sorted(languages),
        issues=issues,
        linter_findings=linter_findings,
        summary=summary,
    )
