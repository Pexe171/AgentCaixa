"""CLI entrypoints for rag_app."""
import json
from pathlib import Path

import typer

from rag_app.config import load_settings
from rag_app.ingest.converters import ensure_docx
from rag_app.ingest.parser_docx import parse_docx_to_blocks

app = typer.Typer(help="rag_app CLI")
INPUT_OPTION = typer.Option(..., "--input", exists=True)


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
            }
            file_handle.write(f"{json.dumps(record, ensure_ascii=False)}\n")

    typer.echo(str(output_path))
