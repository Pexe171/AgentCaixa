"""CLI entrypoints for rag_app."""
from pathlib import Path

import typer

from rag_app.config import load_settings
from rag_app.ingest.converters import ensure_docx

app = typer.Typer(help="rag_app CLI")
INPUT_OPTION = typer.Option(..., "--input", exists=True)


@app.command("ingest")
def ingest(input_path: Path = INPUT_OPTION) -> None:
    """Prepare a document for ingestion (conversion only)."""
    settings = load_settings()
    work_dir = Path(settings.PROCESSED_DIR) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    docx_path = ensure_docx(input_path, work_dir)
    typer.echo(str(docx_path))
