from __future__ import annotations

import base64
import io

from fastapi import APIRouter, Request
from PIL import Image

from app.errors import EMPTY_RESULT, INVALID_IMAGE, SAMError
from app.schemas import (
    Meta,
    Prompt,
    SegmentRequest,
    SegmentResponse,
    TextPointsRequest,
    TextPointsResponse,
)

router = APIRouter()


def _decode_base64_image(image_b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise SAMError(INVALID_IMAGE, f"Could not decode image: {exc}")


@router.post("/v1/sam/segment", response_model=SegmentResponse)
def segment(request: Request, body: SegmentRequest) -> SegmentResponse:
    image = _decode_base64_image(body.image)

    # Run inference
    sam_service = request.app.state.sam_service
    mask_results, point_results = sam_service.predict(
        image,
        body.prompt,
        multimask_output=body.multimask_output,
        max_masks=body.max_masks,
    )

    meta = Meta(
        image_width=image.width,
        image_height=image.height,
        model="sam3",
        prompt_type=body.prompt.prompt_type,
        multimask_output=body.multimask_output,
    )

    if body.output == "points":
        return SegmentResponse(points=point_results, meta=meta)
    return SegmentResponse(masks=mask_results, meta=meta)


@router.post("/v1/sam/segment/text-points", response_model=TextPointsResponse)
def segment_text_points(request: Request, body: TextPointsRequest) -> TextPointsResponse:
    image = _decode_base64_image(body.image)

    sam_service = request.app.state.sam_service
    _mask_results, point_results = sam_service.predict(
        image,
        Prompt(text=body.text),
        multimask_output=False,
        max_masks=1,
    )
    if not point_results:
        raise SAMError(EMPTY_RESULT, "Model returned no points")

    meta = Meta(
        image_width=image.width,
        image_height=image.height,
        model="sam3",
        prompt_type="text",
        multimask_output=False,
    )

    return TextPointsResponse(points=point_results[:1], meta=meta)
