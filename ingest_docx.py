"""Ingestão de documentos .docx com foco em precisão textual.

Este script extrai texto de parágrafos e tabelas (linha por linha),
realiza chunking conservador e salva os chunks em um JSON temporário
para auditoria manual.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph


@dataclass(frozen=True)
class Chunk:
    """Representa um trecho de texto pronto para indexação/auditoria."""

    id: int
    conteudo: str
    tamanho: int


def iterar_blocos(documento: DocxDocument) -> Iterator[Paragraph | Table]:
    """Itera pelos blocos do documento preservando a ordem original.

    A iteração considera parágrafos e tabelas no fluxo real do arquivo,
    evitando perda de contexto estrutural.
    """

    corpo = documento.element.body
    for filho in corpo.iterchildren():
        if isinstance(filho, CT_P):
            yield Paragraph(filho, documento)
        elif isinstance(filho, CT_Tbl):
            yield Table(filho, documento)


def extrair_tabela_linha_a_linha(tabela: Table) -> list[str]:
    """Extrai o conteúdo de uma tabela com granularidade de linha.

    Cada linha é convertida em uma string única, separando células por " | ".
    """

    linhas: list[str] = []

    for linha in tabela.rows:
        celulas: list[str] = []
        for celula in linha.cells:
            texto = extrair_texto_celula(celula)
            celulas.append(texto)

        texto_linha = " | ".join(celulas).strip()
        if texto_linha:
            linhas.append(texto_linha)

    return linhas


def extrair_texto_celula(celula: _Cell) -> str:
    """Extrai texto de uma célula sem perder quebras relevantes."""

    paragrafos = [paragrafo.text for paragrafo in celula.paragraphs]
    # Mantém separação entre parágrafos internos da célula.
    return "\n".join(paragrafos).strip()


def extrair_texto_docx(caminho_docx: Path) -> str:
    """Extrai texto de um arquivo .docx incluindo tabelas."""

    documento = Document(str(caminho_docx))
    linhas_extraidas: list[str] = []

    for bloco in iterar_blocos(documento):
        if isinstance(bloco, Paragraph):
            texto = bloco.text.strip()
            if texto:
                linhas_extraidas.append(texto)
        elif isinstance(bloco, Table):
            linhas_extraidas.extend(extrair_tabela_linha_a_linha(bloco))

    return "\n".join(linhas_extraidas)


def gerar_chunks(texto: str, tamanho_maximo: int = 400, overlap: int = 150) -> list[Chunk]:
    """Cria chunks conservadores com sobreposição para preservar contexto.

    Regras:
    - tamanho máximo por chunk: 400 caracteres (padrão)
    - sobreposição entre chunks: 150 caracteres (padrão)
    """

    if tamanho_maximo <= 0:
        raise ValueError("tamanho_maximo deve ser maior que zero")
    if overlap < 0:
        raise ValueError("overlap não pode ser negativo")
    if overlap >= tamanho_maximo:
        raise ValueError("overlap deve ser menor que tamanho_maximo")

    texto = texto.strip()
    if not texto:
        return []

    chunks: list[Chunk] = []
    inicio = 0
    passo = tamanho_maximo - overlap

    while inicio < len(texto):
        fim = min(inicio + tamanho_maximo, len(texto))
        conteudo = texto[inicio:fim]

        chunks.append(
            Chunk(
                id=len(chunks) + 1,
                conteudo=conteudo,
                tamanho=len(conteudo),
            )
        )

        if fim == len(texto):
            break

        inicio += passo

    return chunks


def salvar_chunks_json(chunks: list[Chunk], caminho_saida: Optional[Path] = None) -> Path:
    """Salva os chunks em JSON em arquivo temporário local para auditoria."""

    if caminho_saida is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho_saida = Path(tempfile.gettempdir()) / f"chunks_auditoria_{timestamp}.json"

    payload = {
        "total_chunks": len(chunks),
        "chunks": [
            {
                "id": chunk.id,
                "tamanho": chunk.tamanho,
                "conteudo": chunk.conteudo,
            }
            for chunk in chunks
        ],
    }

    caminho_saida.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho_saida


def parsear_argumentos() -> argparse.Namespace:
    """Configura e interpreta os argumentos de linha de comando."""

    parser = argparse.ArgumentParser(
        description="Extrai texto de .docx, gera chunks conservadores e salva JSON temporário."
    )
    parser.add_argument("docx", type=Path, help="Caminho para o arquivo .docx de entrada")
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="Caminho opcional do JSON de saída (padrão: arquivo temporário local)",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do script."""

    args = parsear_argumentos()

    if not args.docx.exists() or args.docx.suffix.lower() != ".docx":
        raise FileNotFoundError("Informe um arquivo .docx válido e existente.")

    texto = extrair_texto_docx(args.docx)
    chunks = gerar_chunks(texto=texto, tamanho_maximo=400, overlap=150)
    arquivo_saida = salvar_chunks_json(chunks, args.saida)

    print(f"Extração concluída. Total de chunks: {len(chunks)}")
    print(f"JSON para auditoria: {arquivo_saida}")


if __name__ == "__main__":
    main()
