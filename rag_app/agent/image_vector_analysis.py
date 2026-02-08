"""Análise vetorial nativa de imagens para apoio a IA multimodal local."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


class ImageAnalysisError(ValueError):
    """Erro de validação/execução na análise de imagem."""


try:  # pragma: no branch
    from PIL import Image, ImageFilter, ImageOps
except ImportError:  # pragma: no cover - depende de ambiente
    Image = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ImageVectorResult:
    """Resultado interno de vetorização + métricas de imagem."""

    image_path: str
    width: int
    height: int
    channels: int
    brightness_mean: float
    brightness_std: float
    edge_density: float
    entropy: float
    top_palette: list[tuple[int, int, int]]
    vector: list[float]


def _require_pillow() -> None:
    if Image is None or ImageFilter is None or ImageOps is None:
        raise ImageAnalysisError(
            "Dependência opcional ausente: instale 'pillow' para análise de imagens."
        )


def _normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return values
    return [value / norm for value in values]


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    score = sum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    return max(-1.0, min(1.0, score))


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[int], mean_value: float) -> float:
    if not values:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _entropy(histogram: list[int]) -> float:
    total = sum(histogram)
    if total == 0:
        return 0.0

    entropy_value = 0.0
    for count in histogram:
        if count <= 0:
            continue
        probability = count / total
        entropy_value -= probability * math.log2(probability)
    return float(entropy_value)


def _palette_top_colors(rgb_pixels: list[tuple[int, int, int]], top_n: int = 5) -> list[tuple[int, int, int]]:
    buckets: dict[tuple[int, int, int], int] = {}
    for red, green, blue in rgb_pixels:
        bucket = (red // 16 * 16, green // 16 * 16, blue // 16 * 16)
        buckets[bucket] = buckets.get(bucket, 0) + 1

    ranked = sorted(buckets.items(), key=lambda item: item[1], reverse=True)
    return [color for color, _count in ranked[:top_n]]


def _build_feature_vector(
    brightness_histogram: list[int],
    edge_histogram: list[int],
    rgb_histograms: tuple[list[int], list[int], list[int]],
    width: int,
    height: int,
) -> list[float]:
    total_pixels = max(1, width * height)
    vector: list[float] = []

    for histogram in (brightness_histogram, edge_histogram, *rgb_histograms):
        for value in histogram:
            vector.append(value / total_pixels)

    vector.append(width / max(1, width + height))
    vector.append(height / max(1, width + height))
    return _normalize_vector(vector)


def extract_image_vector(image_path: str) -> ImageVectorResult:
    """Extrai vetor e métricas quantitativas de imagem sem API externa."""

    _require_pillow()
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise ImageAnalysisError("Caminho de imagem inválido ou inexistente.")

    with Image.open(path) as opened_image:  # type: ignore[union-attr]
        rgb_image = opened_image.convert("RGB")

    width, height = rgb_image.size
    if width <= 0 or height <= 0:
        raise ImageAnalysisError("Imagem inválida: dimensões não suportadas.")

    grayscale = ImageOps.grayscale(rgb_image)  # type: ignore[union-attr]
    grayscale_pixels = list(grayscale.getdata())
    brightness_mean = _mean(grayscale_pixels)
    brightness_std = _std(grayscale_pixels, brightness_mean)

    brightness_histogram = grayscale.histogram()
    edge_image = grayscale.filter(ImageFilter.FIND_EDGES)  # type: ignore[union-attr]
    edge_pixels = list(edge_image.getdata())
    edge_histogram = edge_image.histogram()

    strong_edge_pixels = sum(1 for value in edge_pixels if value >= 32)
    edge_density = strong_edge_pixels / len(edge_pixels)

    red_hist, green_hist, blue_hist = rgb_image.split()
    red_histogram = red_hist.histogram()
    green_histogram = green_hist.histogram()
    blue_histogram = blue_hist.histogram()

    palette = _palette_top_colors(list(rgb_image.getdata()))
    vector = _build_feature_vector(
        brightness_histogram=brightness_histogram,
        edge_histogram=edge_histogram,
        rgb_histograms=(red_histogram, green_histogram, blue_histogram),
        width=width,
        height=height,
    )

    return ImageVectorResult(
        image_path=str(path),
        width=width,
        height=height,
        channels=3,
        brightness_mean=round(brightness_mean, 4),
        brightness_std=round(brightness_std, 4),
        edge_density=round(edge_density, 6),
        entropy=round(_entropy(brightness_histogram), 6),
        top_palette=palette,
        vector=vector,
    )


def compare_images(reference_image_path: str, target_image_path: str) -> tuple[ImageVectorResult, ImageVectorResult, float]:
    """Compara duas imagens por similaridade de cosseno no espaço vetorial."""

    reference_result = extract_image_vector(reference_image_path)
    target_result = extract_image_vector(target_image_path)
    similarity = _cosine_similarity(reference_result.vector, target_result.vector)
    similarity_normalized = (similarity + 1.0) / 2.0
    return reference_result, target_result, round(similarity_normalized, 6)
