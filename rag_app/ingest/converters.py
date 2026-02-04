"""Document conversion helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def is_doc(path: Path) -> bool:
    return path.suffix.lower() == ".doc"


def is_docx(path: Path) -> bool:
    return path.suffix.lower() == ".docx"


def ensure_docx(input_path: Path, work_dir: Path) -> Path:
    input_path = Path(input_path)
    work_dir = Path(work_dir)

    if is_docx(input_path):
        return input_path

    if not is_doc(input_path):
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    output_dir = work_dir / "docx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.docx"

    soffice_path = shutil.which("soffice")
    if not soffice_path:
        command = (
            f"soffice --headless --convert-to docx --outdir {output_dir} {input_path}"
        )
        raise RuntimeError(
            "Conversão de .doc para .docx requer o LibreOffice instalado e acessível "
            "no PATH. Comando esperado: "
            f"{command}. "
            "Alternativa: converta manualmente para .docx e informe o caminho do "
            "arquivo .docx."
        )

    try:
        subprocess.run(
            [
                soffice_path,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(output_dir),
                str(input_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Falha ao converter .doc para .docx usando LibreOffice. "
            f"Verifique o comando e tente novamente: "
            f"soffice --headless --convert-to docx --outdir {output_dir} {input_path}"
        ) from exc

    if not output_path.exists():
        raise RuntimeError(
            "Conversão concluída sem gerar o arquivo esperado. "
            f"Arquivo esperado: {output_path}"
        )

    return output_path
