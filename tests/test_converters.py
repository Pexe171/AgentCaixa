from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rag_app.ingest.converters import ensure_docx


def test_ensure_docx_returns_original_for_docx(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.docx"
    input_path.write_text("dummy")
    with patch("rag_app.ingest.converters.shutil.which") as which:
        result = ensure_docx(input_path, tmp_path)
    assert result == input_path
    which.assert_not_called()


def test_ensure_docx_raises_when_soffice_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.doc"
    input_path.write_text("dummy")
    with patch("rag_app.ingest.converters.shutil.which", return_value=None):
        with pytest.raises(RuntimeError) as excinfo:
            ensure_docx(input_path, tmp_path)
    message = str(excinfo.value)
    assert "LibreOffice" in message
    assert "soffice --headless --convert-to docx" in message
    assert "converta manualmente" in message
