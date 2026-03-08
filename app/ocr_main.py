from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.errors import SAMError, request_validation_error_handler, sam_error_handler
from app.ocr_service import OCRService
from app.routes.ocr import router as ocr_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ocr_service = OCRService()
    ocr_service.load()
    app.state.ocr_service = ocr_service
    yield

app = FastAPI(
    title="Paddle OCR Service",
    summary="Standalone OCR API used by sam-finds and external applications.",
    description=(
        "A standalone OCR service for full-page text extraction. "
        "It powers sam-finds via `OCR_SERVICE_URL` and can be consumed directly by other apps "
        "through `/docs` and `/openapi.json`."
    ),
    version="1.0.0",
    openapi_tags=[
        {
            "name": "ocr",
            "description": (
                "Standalone OCR operations for discovering backend capabilities and performing "
                "full-page OCR on images."
            ),
        }
    ],
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_exception_handler(SAMError, sam_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, request_validation_error_handler)  # type: ignore[arg-type]
app.include_router(ocr_router)
