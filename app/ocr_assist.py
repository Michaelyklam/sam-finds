from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from app.errors import EMPTY_RESULT, OCR_ERROR, SAMError
from app.schemas import BoxPrompt, CentroidPoint, MaskRLE, MaskResult, PointResult, Prompt
from app.text_match import OCRMatch, rank_ocr_matches


def _blend_confidence(ocr_score: float, sam_confidence: float) -> float:
    return round((0.7 * ocr_score) + (0.3 * sam_confidence), 4)


def _clamp_ocr_bbox(match: OCRMatch, *, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    det = match.detection
    x1 = max(0, min(image_width - 1, int(det.x1)))
    y1 = max(0, min(image_height - 1, int(det.y1)))
    x2 = max(0, min(image_width, int(det.x2)))
    y2 = max(0, min(image_height, int(det.y2)))
    if x2 <= x1:
        x2 = min(image_width, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height, y1 + 1)
    return x1, y1, x2, y2


def _normalized_polygon(match: OCRMatch, *, image_width: int, image_height: int) -> list[tuple[int, int]]:
    det = match.detection
    if det.polygon and len(det.polygon) >= 3:
        points: list[tuple[int, int]] = []
        for px, py in det.polygon:
            x = max(0, min(image_width - 1, int(px)))
            y = max(0, min(image_height - 1, int(py)))
            points.append((x, y))
        return points

    x1, y1, x2, y2 = _clamp_ocr_bbox(match, image_width=image_width, image_height=image_height)
    return [
        (x1, y1),
        (x2 - 1, y1),
        (x2 - 1, y2 - 1),
        (x1, y2 - 1),
    ]


def _polygon_centroid(points: list[tuple[int, int]]) -> tuple[float, float]:
    # Polygon centroid via shoelace formula; robust fallback for near-degenerate shapes.
    twice_area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        cross = (x1 * y2) - (x2 * y1)
        twice_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    if abs(twice_area) < 1e-6:
        mean_x = sum(p[0] for p in points) / max(1, len(points))
        mean_y = sum(p[1] for p in points) / max(1, len(points))
        return mean_x, mean_y

    area_factor = 1.0 / (3.0 * twice_area)
    return cx * area_factor, cy * area_factor


def _ocr_box_result(match: OCRMatch, *, image_width: int, image_height: int) -> tuple[MaskResult, PointResult]:
    x1, y1, x2, y2 = _clamp_ocr_bbox(match, image_width=image_width, image_height=image_height)
    confidence = round(float(match.score), 4)

    polygon = _normalized_polygon(match, image_width=image_width, image_height=image_height)
    mask_image = Image.new("L", (image_width, image_height), 0)
    ImageDraw.Draw(mask_image).polygon(polygon, fill=1, outline=1)
    binary = np.array(mask_image, dtype=np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary))

    cx, cy = _polygon_centroid(polygon)
    cx = round(max(float(x1), min(float(x2 - 1), cx)), 2)
    cy = round(max(float(y1), min(float(y2 - 1), cy)), 2)

    mask = MaskResult(
        id="0",
        confidence=confidence,
        mask_rle=MaskRLE(
            counts=rle["counts"].decode("utf-8"),
            size=list(rle["size"]),
        ),
    )
    point = PointResult(
        id="0",
        confidence=confidence,
        point=CentroidPoint(x=cx, y=cy),
    )
    return mask, point


def _reindex_results(mask_results, point_results, max_masks: int):
    mask_results.sort(key=lambda m: m.confidence, reverse=True)
    point_results.sort(key=lambda p: p.confidence, reverse=True)

    mask_results = [
        m.model_copy(update={"id": str(i)})
        for i, m in enumerate(mask_results[:max_masks])
    ]
    point_results = [
        p.model_copy(update={"id": str(i)})
        for i, p in enumerate(point_results[:max_masks])
    ]
    return mask_results, point_results


def predict_text_with_ocr_assist(
    *,
    image: Image.Image,
    target_text: str,
    sam_service,
    ocr_service,
    max_masks: int,
    strict_ocr: bool = False,
    use_sam_refine: bool = True,
):
    try:
        detections = ocr_service.detect_text(image)
    except SAMError as exc:
        if strict_ocr or exc.code != OCR_ERROR:
            raise
        return [], [], False

    matches = rank_ocr_matches(
        detections,
        target_text,
        image_width=image.width,
        image_height=image.height,
    )
    if not matches:
        return [], [], False

    mask_results = []
    point_results = []
    for match in matches[:max_masks]:
        if use_sam_refine:
            det = match.detection
            try:
                sub_masks, sub_points = sam_service.predict(
                    image,
                    Prompt(box=BoxPrompt(x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2)),
                    multimask_output=False,
                    max_masks=1,
                )
            except SAMError as exc:
                if exc.code == EMPTY_RESULT:
                    continue
                raise

            if not sub_masks or not sub_points:
                continue

            blended = _blend_confidence(match.score, sub_points[0].confidence)
            mask_results.append(sub_masks[0].model_copy(update={"confidence": blended}))
            point_results.append(sub_points[0].model_copy(update={"confidence": blended}))
        else:
            ocr_mask, ocr_point = _ocr_box_result(
                match,
                image_width=image.width,
                image_height=image.height,
            )
            mask_results.append(ocr_mask)
            point_results.append(ocr_point)

    if not mask_results or not point_results:
        return [], [], False

    mask_results, point_results = _reindex_results(mask_results, point_results, max_masks)
    return mask_results, point_results, True
