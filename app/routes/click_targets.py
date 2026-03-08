from __future__ import annotations
import time

from fastapi import APIRouter, Request

from app.errors import EMPTY_RESULT, TEXT_NOT_FOUND, SAMError
from app.image_utils import decode_base64_image
from app.schemas import (
    BoxPrompt,
    CentroidPoint,
    ClickCandidate,
    ClickTargetsMeta,
    ClickTargetsRequest,
    ClickTargetsResponse,
    Prompt,
    Rect,
)
from app.text_match import rank_ocr_matches

router = APIRouter()


@router.post("/v1/ui/click-targets", response_model=ClickTargetsResponse)
def click_targets(request: Request, body: ClickTargetsRequest) -> ClickTargetsResponse:
    started = time.perf_counter()
    image = decode_base64_image(body.image)
    width, height = image.size

    ocr_service = request.app.state.ocr_service
    sam_service = request.app.state.sam_service

    detections = ocr_service.detect_text(image)
    matches = rank_ocr_matches(
        detections,
        body.target_text,
        image_width=width,
        image_height=height,
    )
    if not matches:
        raise SAMError(TEXT_NOT_FOUND, f"No OCR text matched target_text='{body.target_text}'")

    selected = matches[:body.max_candidates]
    candidates: list[ClickCandidate] = []
    for idx, match in enumerate(selected):
        det = match.detection
        rect = Rect(x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2)

        # Default fallback is OCR box center.
        center_x = round((det.x1 + det.x2) / 2.0, 2)
        center_y = round((det.y1 + det.y2) / 2.0, 2)
        point = CentroidPoint(x=center_x, y=center_y)
        sam_confidence: float | None = None

        if body.use_sam_refine:
            try:
                _mask_results, point_results = sam_service.predict(
                    image,
                    Prompt(box=BoxPrompt(x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2)),
                    multimask_output=False,
                    max_masks=1,
                )
                if point_results:
                    point = point_results[0].point
                    sam_confidence = point_results[0].confidence
            except SAMError as exc:
                # Empty result is expected occasionally for thin text boxes.
                if exc.code != EMPTY_RESULT:
                    raise

        score = match.score
        if sam_confidence is not None:
            score = round((0.7 * match.score) + (0.3 * sam_confidence), 4)

        candidates.append(
            ClickCandidate(
                id=str(idx),
                point=point,
                score=score,
                ocr_text=det.text,
                ocr_confidence=round(det.confidence, 4),
                sam_confidence=sam_confidence,
                bbox=rect,
            )
        )

    candidates.sort(
        key=lambda c: (c.score, c.sam_confidence or 0.0, c.ocr_confidence),
        reverse=True,
    )

    latency_ms = int(round((time.perf_counter() - started) * 1000))
    meta = ClickTargetsMeta(
        pipeline="ocr+sam" if body.use_sam_refine else "ocr",
        target_text=body.target_text,
        num_ocr_hits=len(matches),
        num_candidates=len(candidates),
        latency_ms=latency_ms,
    )
    return ClickTargetsResponse(candidates=candidates, meta=meta)
