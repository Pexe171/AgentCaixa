"""Parse .docx files into ordered blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


@dataclass(frozen=True)
class Block:
    type: Literal["paragraph", "table"]
    text: str
    order: int


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _serialize_table(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [_normalize_text(cell.text) for cell in row.cells]
        if not any(cells):
            continue
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _iter_block_items(document: Document):
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def parse_docx_to_blocks(docx_path: Path) -> list[Block]:
    doc = Document(docx_path)
    blocks: list[Block] = []
    order = 1

    for item in _iter_block_items(doc):
        if isinstance(item, Paragraph):
            text = _normalize_text(item.text)
            if not text:
                continue
            blocks.append(Block(type="paragraph", text=text, order=order))
        else:
            text = _serialize_table(item)
            if not text:
                continue
            blocks.append(Block(type="table", text=text, order=order))
        order += 1

    return blocks
