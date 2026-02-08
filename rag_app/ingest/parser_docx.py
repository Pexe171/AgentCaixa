"""Parse .docx files into ordered blocks, incluindo OCR de imagens."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

from docx import Document
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

try:
    from PIL import Image
except Exception:  # pragma: no cover - dependência opcional
    Image = None

try:
    import pytesseract
except Exception:  # pragma: no cover - dependência opcional
    pytesseract = None


@dataclass(frozen=True)
class Block:
    type: Literal["paragraph", "table", "image"]
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


def _extract_image_rel_ids(paragraph: Paragraph) -> list[str]:
    rel_ids: list[str] = []
    for run in paragraph.runs:
        blips = run._element.xpath(".//a:blip")
        for blip in blips:
            rel_id = blip.get(qn("r:embed"))
            if rel_id:
                rel_ids.append(rel_id)
    return rel_ids


def _extract_ocr_text(document: Document, rel_id: str) -> str:
    image_part = document.part.related_parts.get(rel_id)
    if image_part is None:
        return ""

    if Image is None or pytesseract is None:
        return "[OCR indisponível: instale pillow + pytesseract]"

    with Image.open(BytesIO(image_part.blob)) as image:
        return _normalize_text(pytesseract.image_to_string(image))


def parse_docx_to_blocks(docx_path: Path) -> list[Block]:
    doc = Document(docx_path)
    blocks: list[Block] = []
    order = 1

    for item in _iter_block_items(doc):
        if isinstance(item, Paragraph):
            text = _normalize_text(item.text)
            if text:
                blocks.append(Block(type="paragraph", text=text, order=order))
                order += 1

            for rel_id in _extract_image_rel_ids(item):
                ocr_text = _extract_ocr_text(doc, rel_id)
                if not ocr_text:
                    continue
                blocks.append(Block(type="image", text=ocr_text, order=order))
                order += 1
        else:
            text = _serialize_table(item)
            if not text:
                continue
            blocks.append(Block(type="table", text=text, order=order))
            order += 1

    return blocks
