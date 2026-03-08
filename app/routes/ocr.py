from __future__ import annotations

from fastapi import APIRouter, Request

from app.errors import INVALID_IMAGE, INVALID_REQUEST, OCR_ERROR
from app.image_utils import decode_base64_image
from app.ocr_service import OCRDetection
from app.schemas import (
    ErrorResponse,
    OCRBackendResponse,
    OCRDetectionResult,
    OCRPageMeta,
    OCRPageRequest,
    OCRPageResponse,
    PolygonPoint,
    Rect,
)

_INVALID_IMAGE_ERROR = {
    "model": ErrorResponse,
    "description": "The image could not be decoded from the provided base64 PNG or JPEG bytes.",
    "content": {
        "application/json": {
            "example": {
                "error": {
                    "code": INVALID_IMAGE,
                    "message": "Could not decode image: Incorrect padding",
                    "hint": "Send `image` as base64-encoded PNG/JPEG bytes.",
                }
            }
        }
    },
}

_OCR_ERROR = {
    "model": ErrorResponse,
    "description": "The OCR backend was unavailable or failed during initialization or inference.",
    "content": {
        "application/json": {
            "example": {
                "error": {
                    "code": OCR_ERROR,
                    "message": "OCR inference failed: backend unavailable",
                    "hint": "If targeting visual objects, retry with `text_mode=visual` to bypass OCR.",
                }
            }
        }
    },
}

_INVALID_REQUEST_ERROR = {
    "model": ErrorResponse,
    "description": "The request body failed schema validation.",
    "content": {
        "application/json": {
            "example": {
                "error": {
                    "code": INVALID_REQUEST,
                    "message": "Invalid request payload",
                    "hint": "Fix request schema fields. `text_mode` is required: use `visual` for object descriptions or `screen_text` for on-screen text labels.",
                    "details": [
                        {
                            "type": "missing",
                            "loc": ["body", "image"],
                            "msg": "Field required",
                            "input": {},
                        }
                    ],
                }
            }
        }
    },
}

router = APIRouter(tags=["ocr"])


def _serialize_detection(
    detection: OCRDetection,
    *,
    idx: int,
    include_polygons: bool,
) -> OCRDetectionResult:
    polygon = None
    if include_polygons and detection.polygon:
        polygon = [PolygonPoint(x=x, y=y) for x, y in detection.polygon]

    return OCRDetectionResult(
        id=str(idx),
        text=detection.text,
        confidence=round(float(detection.confidence), 4),
        bbox=Rect(x1=detection.x1, y1=detection.y1, x2=detection.x2, y2=detection.y2),
        polygon=polygon,
    )


@router.get(
    "/v1/ocr/backend",
    response_model=OCRBackendResponse,
    summary="Get OCR backend details",
    description=(
        "Returns the currently active OCR backend, execution device, configured language set, "
        "and detector side length limit. External consumers can use this endpoint as a startup "
        "diagnostic and capability check before sending page OCR requests."
    ),
    operation_id="getOcrBackend",
    response_description="OCR runtime information for the currently active backend.",
    responses={500: _OCR_ERROR},
)
def ocr_backend(request: Request) -> OCRBackendResponse:
    ocr_service = request.app.state.ocr_service
    backend = ocr_service.describe_backend()
    return OCRBackendResponse(**backend)


@router.post(
    "/v1/ocr/page",
    response_model=OCRPageResponse,
    summary="Run full-page OCR",
    description=(
        "Performs full-page OCR on a base64-encoded PNG or JPEG image. The response includes a "
        "convenience `text` field plus structured detections with confidence scores, bounding boxes, "
        "and optional polygon geometry for downstream automation."
    ),
    operation_id="postOcrPage",
    response_description="Full-page OCR results for the submitted image.",
    responses={
        400: _INVALID_IMAGE_ERROR,
        422: _INVALID_REQUEST_ERROR,
        500: _OCR_ERROR,
    },
)
def ocr_page(request: Request, body: OCRPageRequest) -> OCRPageResponse:
    image = decode_base64_image(body.image)
    ocr_service = request.app.state.ocr_service
    detections = ocr_service.detect_text(image)

    results = [
        _serialize_detection(det, idx=idx, include_polygons=body.include_polygons)
        for idx, det in enumerate(detections)
    ]
    full_text = "\n".join(det.text for det in detections)
    backend = ocr_service.describe_backend()

    return OCRPageResponse(
        text=full_text,
        detections=results,
        meta=OCRPageMeta(
            image_width=image.width,
            image_height=image.height,
            backend=str(backend["backend"]),
            backend_device=str(backend["backend_device"]),
            lang=str(backend["lang"]),
            num_detections=len(results),
        ),
    )
