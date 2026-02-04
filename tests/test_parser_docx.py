from pathlib import Path

from docx import Document

from rag_app.ingest.parser_docx import parse_docx_to_blocks


def _create_sample_docx(path: Path) -> None:
    doc = Document()
    doc.add_paragraph("Primeiro parágrafo.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A1"
    table.cell(0, 1).text = "B1"
    table.cell(1, 0).text = "A2"
    table.cell(1, 1).text = "B2"
    doc.add_paragraph("Segundo parágrafo.")
    doc.save(path)


def test_parse_docx_to_blocks_preserves_order(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    _create_sample_docx(docx_path)

    blocks = parse_docx_to_blocks(docx_path)

    assert len(blocks) == 3
    assert [block.type for block in blocks] == [
        "paragraph",
        "table",
        "paragraph",
    ]
    assert [block.order for block in blocks] == [1, 2, 3]
    assert " | " in blocks[1].text
