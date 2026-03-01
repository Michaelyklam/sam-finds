from fastapi import Request
from fastapi.responses import JSONResponse


class SAMError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code


INVALID_IMAGE = "INVALID_IMAGE"
INVALID_PROMPT = "INVALID_PROMPT"
MODEL_ERROR = "MODEL_ERROR"
EMPTY_RESULT = "EMPTY_RESULT"


async def sam_error_handler(_request: Request, exc: SAMError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
