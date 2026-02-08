"""Varredura estática leve para suporte operacional do agente."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rag_app.agent.schemas import AgentScanResponse, ScanIssue

_SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".sh": "shell",
}

_SECRET_HINTS = ("api_key", "secret", "token", "senha", "password")
_DEBUG_HINTS = ("console.log(", "print(", "debugger", "pdb.set_trace")
_ERROR_HINTS = ("except:", "catch (e)", "catch(e)")
_COMMENTED_CODE_HINTS = ("// if (", "# if ", "/* TODO", "# TODO")


def _is_hidden(path: Path, root: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part.startswith(".") for part in relative_parts)


def _iter_files(folder: Path, include_hidden: bool, max_files: int) -> list[Path]:
    files: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if not include_hidden and _is_hidden(path, folder):
            continue
        files.append(path)
        if len(files) >= max_files:
            break
    return files


def _detect_issues(file_path: Path, root: Path) -> list[ScanIssue]:
    issues: list[ScanIssue] = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return issues

    for line_number, line in enumerate(content.splitlines(), start=1):
        lower_line = line.lower()

        if any(hint in lower_line for hint in _SECRET_HINTS) and "=" in line:
            issues.append(
                ScanIssue(
                    file_path=str(file_path.relative_to(root)),
                    line_number=line_number,
                    severity="alta",
                    category="possivel-segredo",
                    message="Possível segredo sensível encontrado no código.",
                    suggestion="Mova segredo para variável de ambiente e remova do repositório.",
                )
            )

        if any(hint in line for hint in _DEBUG_HINTS):
            issues.append(
                ScanIssue(
                    file_path=str(file_path.relative_to(root)),
                    line_number=line_number,
                    severity="media",
                    category="debug-residual",
                    message="Trecho de debug residual detectado.",
                    suggestion="Remova logs de depuração antes de subir para produção.",
                )
            )

        if any(hint in lower_line for hint in _ERROR_HINTS):
            issues.append(
                ScanIssue(
                    file_path=str(file_path.relative_to(root)),
                    line_number=line_number,
                    severity="media",
                    category="tratamento-de-erros",
                    message="Tratamento de erro genérico pode esconder causa raiz.",
                    suggestion="Capture exceções específicas e registre contexto do erro.",
                )
            )

        if any(hint.lower() in lower_line for hint in _COMMENTED_CODE_HINTS):
            issues.append(
                ScanIssue(
                    file_path=str(file_path.relative_to(root)),
                    line_number=line_number,
                    severity="baixa",
                    category="codigo-comentado",
                    message="Comentário sugere código legado ou desativado.",
                    suggestion="Remova bloco morto ou converta em issue/tarefa documentada.",
                )
            )

    if not issues:
        issues.append(
            ScanIssue(
                file_path=str(file_path.relative_to(root)),
                line_number=1,
                severity="baixa",
                category="boas-praticas",
                message="Nenhum problema relevante identificado no arquivo.",
                suggestion="Manter revisão periódica e cobertura de testes.",
            )
        )

    return issues


def _run_linters(folder: Path, languages: set[str]) -> list[str]:
    findings: list[str] = []
    commands: list[tuple[str, list[str]]] = []

    if "python" in languages:
        commands.append(("python -m py_compile", ["python", "-m", "py_compile", *[str(p) for p in folder.rglob("*.py")]]))

    for label, command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=folder,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                findings.append(f"{label}: ok")
            else:
                findings.append(f"{label}: falhou - {result.stderr.strip() or result.stdout.strip()}")
        except OSError as exc:
            findings.append(f"{label}: não executado ({exc})")

    if not findings:
        findings.append("Nenhum linter aplicável para as linguagens encontradas.")

    return findings


def scan_folder(
    folder_path: str,
    include_hidden: bool = False,
    max_files: int = 400,
    run_linters: bool = False,
) -> AgentScanResponse:
    """Varre uma pasta inteira e retorna diagnóstico resumido."""

    root = Path(folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError("Pasta informada não existe ou não é um diretório válido.")

    files = _iter_files(root, include_hidden=include_hidden, max_files=max_files)
    languages: set[str] = set()
    issues: list[ScanIssue] = []

    for file_path in files:
        language = _SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())
        if language:
            languages.add(language)
        issues.extend(_detect_issues(file_path=file_path, root=root))

    linter_findings = _run_linters(root, languages) if run_linters else []
    summary = (
        f"Arquivos analisados: {len(files)}. "
        f"Linguagens detectadas: {', '.join(sorted(languages)) or 'nenhuma'}. "
        f"Achados: {len(issues)}. "
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
