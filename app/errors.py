from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class SAMError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        hint: str | None = None,
        details: list[dict[str, Any]] | None = None,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.hint = hint
        self.details = details


INVALID_IMAGE = "INVALID_IMAGE"
INVALID_PROMPT = "INVALID_PROMPT"
MODEL_ERROR = "MODEL_ERROR"
EMPTY_RESULT = "EMPTY_RESULT"
TEXT_NOT_FOUND = "TEXT_NOT_FOUND"
AMBIGUOUS_TARGET = "AMBIGUOUS_TARGET"
OCR_ERROR = "OCR_ERROR"
INVALID_REQUEST = "INVALID_REQUEST"


def _default_hint_for_code(code: str) -> str | None:
    hints: dict[str, str] = {
        INVALID_IMAGE: "Send `image` as base64-encoded PNG/JPEG bytes.",
        INVALID_PROMPT: "Provide exactly one of `prompt.text`, `prompt.box`, or `prompt.points`.",
        EMPTY_RESULT: "Try a more specific prompt. Use `text_mode=screen_text` for on-screen labels, otherwise `text_mode=visual`.",
        TEXT_NOT_FOUND: "Use exact visible label text and `text_mode=screen_text`.",
        OCR_ERROR: "If targeting visual objects, retry with `text_mode=visual` to bypass OCR.",
    }
    return hints.get(code)


async def sam_error_handler(_request: Request, exc: SAMError) -> JSONResponse:
    payload: dict[str, Any] = {"code": exc.code, "message": exc.message}
    hint = exc.hint or _default_hint_for_code(exc.code)
    if hint is not None:
        payload["hint"] = hint
    if exc.details is not None:
        payload["details"] = exc.details
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": payload},
    )


async def request_validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    hint = (
        "Fix request schema fields. `text_mode` is required: use `visual` for object descriptions "
        "or `screen_text` for on-screen text labels."
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": INVALID_REQUEST,
                "message": "Invalid request payload",
                "hint": hint,
                "details": errors,
            }
        },
    )
