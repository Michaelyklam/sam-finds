from __future__ import annotations

from fastapi import APIRouter, Request

from app.image_utils import decode_base64_image
from app.ocr_assist import predict_text_with_ocr_assist
from app.schemas import Meta, SegmentRequest, SegmentResponse

router = APIRouter()


def _should_use_ocr_for_ui_text(*, text_mode: str) -> bool:
    return text_mode == "screen_text"


@router.post("/v1/ui/segment", response_model=SegmentResponse)
def ui_segment(request: Request, body: SegmentRequest) -> SegmentResponse:
    image = decode_base64_image(body.image)
    sam_service = request.app.state.sam_service

    used_ocr_assist = False
    mask_results = []
    point_results = []

    if body.prompt.text is not None and _should_use_ocr_for_ui_text(text_mode=body.text_mode):
        ocr_service = request.app.state.ocr_service
        mask_results, point_results, used_ocr_assist = predict_text_with_ocr_assist(
            image=image,
            target_text=body.prompt.text,
            sam_service=sam_service,
            ocr_service=ocr_service,
            max_masks=body.max_masks,
            use_sam_refine=False,
        )

    if not mask_results and not point_results:
        # Fallback to baseline SAM behavior when OCR does not produce useful anchors.
        mask_results, point_results = sam_service.predict(
            image,
            body.prompt,
            multimask_output=body.multimask_output,
            max_masks=body.max_masks,
        )

    if body.output == "points":
        point_results.sort(key=lambda p: p.confidence, reverse=True)
        point_results = [
            p.model_copy(update={"id": str(i)})
            for i, p in enumerate(point_results[: body.max_masks])
        ]
    else:
        mask_results.sort(key=lambda m: m.confidence, reverse=True)
        mask_results = [
            m.model_copy(update={"id": str(i)})
            for i, m in enumerate(mask_results[: body.max_masks])
        ]

    meta = Meta(
        image_width=image.width,
        image_height=image.height,
        model="sam3+ocr" if used_ocr_assist else "sam3",
        prompt_type=body.prompt.prompt_type,
        multimask_output=False if used_ocr_assist and body.prompt.text is not None else body.multimask_output,
    )

    if body.output == "points":
        return SegmentResponse(points=point_results, meta=meta)
    return SegmentResponse(masks=mask_results, meta=meta)
