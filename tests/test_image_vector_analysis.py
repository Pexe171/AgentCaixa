from pathlib import Path

import pytest

from rag_app.agent.image_vector_analysis import (
    ImageAnalysisError,
    compare_images,
    extract_image_vector,
)

pytest.importorskip("PIL")
from PIL import Image


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (32, 32), color=color)
    image.save(path)


def test_extract_image_vector_returns_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _create_image(image_path, (120, 180, 240))

    result = extract_image_vector(str(image_path))

    assert result.width == 32
    assert result.height == 32
    assert result.channels == 3
    assert result.edge_density >= 0.0
    assert result.entropy >= 0.0
    assert len(result.vector) > 100
    assert len(result.top_palette) > 0


def test_compare_images_returns_similarity(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _create_image(image_a, (32, 128, 200))
    _create_image(image_b, (32, 128, 200))

    _ref, _target, similarity = compare_images(str(image_a), str(image_b))

    assert similarity >= 0.95


def test_extract_image_vector_invalid_path_raises() -> None:
    with pytest.raises(ImageAnalysisError):
        extract_image_vector("/nao/existe/imagem.png")
