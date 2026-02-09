"""CLI entrypoints for rag_app."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from rag_app.agent.schemas import AgentScanRequest
from rag_app.agent.service import AgentService
from rag_app.config import load_settings
from rag_app.ingest.converters import ensure_docx
from rag_app.ingest.parser_docx import parse_docx_to_blocks

app = typer.Typer(help="rag_app CLI")
INPUT_OPTION = typer.Option(..., "--input", exists=True)
SCAN_FOLDER_OPTION = typer.Option(..., "--folder", exists=True, file_okay=False)
SCAN_INCLUDE_HIDDEN_OPTION = typer.Option(False, "--include-hidden")
SCAN_MAX_FILES_OPTION = typer.Option(400, "--max-files", min=1, max=5000)


@app.command("ingest")
def ingest(input_path: Path = INPUT_OPTION) -> None:
    """Prepare a document for ingestion (conversion + blocks)."""
    settings = load_settings()
    work_dir = Path(settings.PROCESSED_DIR) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    docx_path = ensure_docx(input_path, work_dir)
    blocks = parse_docx_to_blocks(docx_path)

    processed_dir = Path(settings.PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "blocks.jsonl"
    with output_path.open("w", encoding="utf-8") as file_handle:
        for block in blocks:
            record = {
                "type": block.type,
                "text": block.text,
                "order": block.order,
                "source_file": str(input_path),
                "file_name": block.file_name,
                "created_at": block.created_at,
                "section": block.section,
            }
            file_handle.write(f"{json.dumps(record, ensure_ascii=False)}\n")

    typer.echo(str(output_path))


@app.command("scan")
def scan(
    folder: Path = SCAN_FOLDER_OPTION,
    include_hidden: bool = SCAN_INCLUDE_HIDDEN_OPTION,
    max_files: int = SCAN_MAX_FILES_OPTION,
) -> None:
    """Varre uma pasta inteira e retorna relatório de problemas comuns."""

    settings = load_settings()
    service = AgentService(settings=settings)
    result = service.scan_codebase(
        AgentScanRequest(
            folder_path=str(folder),
            include_hidden=include_hidden,
            max_files=max_files,
        )
    )
    typer.echo(result.model_dump_json(indent=2, ensure_ascii=False))
