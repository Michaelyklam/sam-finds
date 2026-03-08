from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TextMode = Literal["visual", "screen_text"]


class PointPrompt(BaseModel):
    x: int
    y: int
    label: Literal[0, 1]


class BoxPrompt(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Prompt(BaseModel):
    points: list[PointPrompt] | None = None
    box: BoxPrompt | None = None
    text: str | None = None

    @model_validator(mode="after")
    def exactly_one_prompt(self) -> Prompt:
        set_fields = sum(
            v is not None for v in [self.points, self.box, self.text]
        )
        if set_fields != 1:
            raise ValueError(
                "Exactly one of 'points', 'box', or 'text' must be provided"
            )
        return self

    @property
    def prompt_type(self) -> str:
        if self.points is not None:
            return "points"
        if self.box is not None:
            return "box"
        return "text"


class SegmentRequest(BaseModel):
    image: str  # base64 PNG/JPEG
    prompt: Prompt
    text_mode: TextMode
    multimask_output: bool = True
    max_masks: int = 3
    output: Literal["masks", "points"] = "masks"


class TextPointsRequest(BaseModel):
    image: str  # base64 PNG/JPEG
    text: str
    text_mode: TextMode

    @field_validator("text")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must be non-empty")
        return normalized


class ClickTargetsRequest(BaseModel):
    image: str  # base64 PNG/JPEG
    target_text: str
    max_candidates: int = Field(default=3, ge=1, le=5)
    use_sam_refine: bool = True

    @field_validator("target_text")
    @classmethod
    def non_empty_target_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("target_text must be non-empty")
        return normalized


class MaskRLE(BaseModel):
    counts: str
    size: list[int]


class MaskResult(BaseModel):
    id: str
    confidence: float
    mask_rle: MaskRLE


class CentroidPoint(BaseModel):
    x: float
    y: float


class PointResult(BaseModel):
    id: str
    confidence: float
    point: CentroidPoint


class Meta(BaseModel):
    image_width: int
    image_height: int
    model: str
    prompt_type: str
    multimask_output: bool


class TextPointsResponse(BaseModel):
    points: list[PointResult]
    meta: Meta


class Rect(BaseModel):
    x1: int = Field(description="Left edge of the axis-aligned bounding box in pixels.", examples=[412])
    y1: int = Field(description="Top edge of the axis-aligned bounding box in pixels.", examples=[1110])
    x2: int = Field(description="Right edge of the axis-aligned bounding box in pixels.", examples=[560])
    y2: int = Field(description="Bottom edge of the axis-aligned bounding box in pixels.", examples=[1234])


class ClickCandidate(BaseModel):
    id: str
    point: CentroidPoint
    score: float
    ocr_text: str
    ocr_confidence: float
    sam_confidence: float | None = None
    bbox: Rect


class ClickTargetsMeta(BaseModel):
    pipeline: str
    target_text: str
    num_ocr_hits: int
    num_candidates: int
    latency_ms: int


class ClickTargetsResponse(BaseModel):
    candidates: list[ClickCandidate]
    meta: ClickTargetsMeta


class SegmentResponse(BaseModel):
    masks: list[MaskResult] | None = None
    points: list[PointResult] | None = None
    meta: Meta


class OCRPageRequest(BaseModel):
    image: str = Field(
        description="Base64-encoded PNG or JPEG bytes for the source image.",
        examples=["iVBORw0KGgoAAAANSUhEUgAA..."],
    )
    include_polygons: bool = Field(
        default=True,
        description="Whether to include polygon geometry for each OCR detection in the response.",
        examples=[True],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": "iVBORw0KGgoAAAANSUhEUgAA...",
                "include_polygons": True,
            }
        }
    )


class PolygonPoint(BaseModel):
    x: int = Field(description="Horizontal pixel coordinate for a polygon vertex.", examples=[412])
    y: int = Field(description="Vertical pixel coordinate for a polygon vertex.", examples=[1110])


class OCRDetectionResult(BaseModel):
    id: str = Field(description="Stable identifier within the response payload.", examples=["0"])
    text: str = Field(description="Recognized text for this detection.", examples=["View Leaderboard"])
    confidence: float = Field(description="OCR confidence score for the recognized text.", examples=[0.96])
    bbox: Rect = Field(description="Axis-aligned bounding box enclosing the text region.")
    polygon: list[PolygonPoint] | None = Field(
        default=None,
        description="Polygon geometry for the text region when `include_polygons=true`.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "0",
                "text": "View Leaderboard",
                "confidence": 0.96,
                "bbox": {"x1": 412, "y1": 1110, "x2": 560, "y2": 1234},
                "polygon": [
                    {"x": 412, "y": 1110},
                    {"x": 560, "y": 1110},
                    {"x": 560, "y": 1234},
                    {"x": 412, "y": 1234},
                ],
            }
        }
    )


class OCRPageMeta(BaseModel):
    image_width: int = Field(description="Width of the decoded source image in pixels.", examples=[1200])
    image_height: int = Field(description="Height of the decoded source image in pixels.", examples=[1602])
    backend: str = Field(description="Active OCR backend serving the request.", examples=["paddle"])
    backend_device: str = Field(description="Execution device for the active OCR backend.", examples=["gpu"])
    lang: str = Field(description="Configured OCR language set.", examples=["en"])
    num_detections: int = Field(description="Number of OCR detections returned.", examples=[2])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_width": 1200,
                "image_height": 1602,
                "backend": "paddle",
                "backend_device": "gpu",
                "lang": "en",
                "num_detections": 2,
            }
        }
    )


class OCRPageResponse(BaseModel):
    text: str = Field(
        description="Convenience newline-joined text extracted from the page. Use `detections` for structured downstream logic.",
        examples=["View Leaderboard\nSettings"],
    )
    detections: list[OCRDetectionResult] = Field(
        description="Structured OCR detections for each text region recognized on the page."
    )
    meta: OCRPageMeta = Field(description="Metadata about the OCR request and active backend.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "View Leaderboard\nSettings",
                "detections": [
                    {
                        "id": "0",
                        "text": "View Leaderboard",
                        "confidence": 0.96,
                        "bbox": {"x1": 412, "y1": 1110, "x2": 560, "y2": 1234},
                        "polygon": [
                            {"x": 412, "y": 1110},
                            {"x": 560, "y": 1110},
                            {"x": 560, "y": 1234},
                            {"x": 412, "y": 1234},
                        ],
                    }
                ],
                "meta": {
                    "image_width": 1200,
                    "image_height": 1602,
                    "backend": "paddle",
                    "backend_device": "gpu",
                    "lang": "en",
                    "num_detections": 2,
                },
            }
        }
    )


class OCRBackendResponse(BaseModel):
    backend: str = Field(description="Active OCR backend name.", examples=["paddle"])
    backend_device: str = Field(description="Execution device for the active OCR backend.", examples=["gpu"])
    lang: str = Field(description="Configured OCR language set.", examples=["en"])
    det_limit_side_len: int = Field(
        description="Maximum image side length used by the OCR text detector before internal resizing.",
        examples=[1920],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "backend": "paddle",
                "backend_device": "gpu",
                "lang": "en",
                "det_limit_side_len": 1920,
            }
        }
    )


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.", examples=["INVALID_IMAGE"])
    message: str = Field(description="Human-readable error message.", examples=["Could not decode image: Incorrect padding"])
    hint: str | None = Field(
        default=None,
        description="Optional remediation hint for clients.",
        examples=["Send `image` as base64-encoded PNG/JPEG bytes."],
    )
    details: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional structured validation or debugging details.",
        examples=[[{"type": "missing", "loc": ["body", "image"], "msg": "Field required"}]],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "INVALID_IMAGE",
                "message": "Could not decode image: Incorrect padding",
                "hint": "Send `image` as base64-encoded PNG/JPEG bytes.",
            }
        }
    )


class ErrorResponse(BaseModel):
    error: ErrorDetail = Field(description="Wrapped error payload.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "INVALID_IMAGE",
                    "message": "Could not decode image: Incorrect padding",
                    "hint": "Send `image` as base64-encoded PNG/JPEG bytes.",
                }
            }
        }
    )
