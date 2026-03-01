from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


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
    mask: str | None = None  # base64-encoded mask

    @model_validator(mode="after")
    def exactly_one_prompt(self) -> Prompt:
        set_fields = sum(
            v is not None for v in [self.points, self.box, self.mask]
        )
        if set_fields != 1:
            raise ValueError(
                "Exactly one of 'points', 'box', or 'mask' must be provided"
            )
        return self

    @property
    def prompt_type(self) -> str:
        if self.points is not None:
            return "points"
        if self.box is not None:
            return "box"
        return "mask"


class SegmentRequest(BaseModel):
    image: str  # base64 PNG/JPEG
    prompt: Prompt
    multimask_output: bool = True
    max_masks: int = 3


class MaskRLE(BaseModel):
    counts: str
    size: list[int]


class MaskResult(BaseModel):
    id: str
    confidence: float
    mask_rle: MaskRLE


class Meta(BaseModel):
    image_width: int
    image_height: int
    model: str
    prompt_type: str
    multimask_output: bool


class SegmentResponse(BaseModel):
    masks: list[MaskResult]
    meta: Meta


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
