from __future__ import annotations

from fastapi import APIRouter, Request

from app.errors import EMPTY_RESULT, SAMError
from app.image_utils import decode_base64_image
from app.ocr_assist import predict_text_with_ocr_assist
from app.schemas import (
    Meta,
    Prompt,
    SegmentRequest,
    SegmentResponse,
    TextPointsRequest,
    TextPointsResponse,
)

router = APIRouter()


def _should_use_ocr_for_segment_text(*, text_mode: str) -> bool:
    return text_mode == "screen_text"


@router.post("/v1/sam/segment", response_model=SegmentResponse)
def segment(request: Request, body: SegmentRequest) -> SegmentResponse:
    image = decode_base64_image(body.image)

    sam_service = request.app.state.sam_service
    used_ocr_assist = False
    mask_results = []
    point_results = []

    if body.prompt.text is not None and _should_use_ocr_for_segment_text(text_mode=body.text_mode):
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
        mask_results, point_results = sam_service.predict(
            image,
            body.prompt,
            multimask_output=body.multimask_output,
            max_masks=body.max_masks,
        )

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


@router.post("/v1/sam/segment/text-points", response_model=TextPointsResponse)
def segment_text_points(request: Request, body: TextPointsRequest) -> TextPointsResponse:
    image = decode_base64_image(body.image)

    sam_service = request.app.state.sam_service
    point_results = []
    used_ocr_assist = False

    if _should_use_ocr_for_segment_text(text_mode=body.text_mode):
        ocr_service = request.app.state.ocr_service
        _mask_results, point_results, used_ocr_assist = predict_text_with_ocr_assist(
            image=image,
            target_text=body.text,
            sam_service=sam_service,
            ocr_service=ocr_service,
            max_masks=1,
            use_sam_refine=False,
        )

    if not point_results:
        _mask_results, point_results = sam_service.predict(
            image,
            Prompt(text=body.text),
            multimask_output=False,
            max_masks=1,
        )
        used_ocr_assist = False

    if not point_results:
        raise SAMError(EMPTY_RESULT, "Model returned no points")

    meta = Meta(
        image_width=image.width,
        image_height=image.height,
        model="sam3+ocr" if used_ocr_assist else "sam3",
        prompt_type="text",
        multimask_output=False,
    )

    return TextPointsResponse(points=point_results[:1], meta=meta)
