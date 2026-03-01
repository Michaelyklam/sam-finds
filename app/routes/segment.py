from __future__ import annotations

import base64
import io

from fastapi import APIRouter, Request
from PIL import Image

from app.errors import INVALID_IMAGE, SAMError
from app.schemas import Meta, SegmentRequest, SegmentResponse

router = APIRouter()


@router.post("/v1/sam/segment", response_model=SegmentResponse)
async def segment(request: Request, body: SegmentRequest) -> SegmentResponse:
    # Decode image
    try:
        image_bytes = base64.b64decode(body.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise SAMError(INVALID_IMAGE, f"Could not decode image: {exc}")

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
