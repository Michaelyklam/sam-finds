from __future__ import annotations

from PIL import Image

from app.errors import EMPTY_RESULT, OCR_ERROR, SAMError
from app.schemas import BoxPrompt, Prompt
from app.text_match import rank_ocr_matches


def _blend_confidence(ocr_score: float, sam_confidence: float) -> float:
    return round((0.7 * ocr_score) + (0.3 * sam_confidence), 4)


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

    if not mask_results or not point_results:
        return [], [], False

    mask_results, point_results = _reindex_results(mask_results, point_results, max_masks)
    return mask_results, point_results, True
