from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import os

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.errors import SAMError, request_validation_error_handler, sam_error_handler
from app.ocr_service import OCRService
from app.ocr_remote_service import RemoteOCRService
from app.routes.click_targets import router as click_targets_router
from app.routes.segment import router as segment_router
from app.routes.ui_segment import router as ui_segment_router
from app.sam_service import SAMService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    sam_service = SAMService()
    sam_service.load()
    app.state.sam_service = sam_service

    ocr_service_url = os.getenv("OCR_SERVICE_URL")
    ocr_service = RemoteOCRService(ocr_service_url) if ocr_service_url else OCRService()
    ocr_service.load()
    app.state.ocr_service = ocr_service
    yield
    close = getattr(ocr_service, "close", None)
    if callable(close):
        close()


app = FastAPI(title="SAM Finds", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_exception_handler(SAMError, sam_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, request_validation_error_handler)  # type: ignore[arg-type]
app.include_router(segment_router)
app.include_router(ui_segment_router)
app.include_router(click_targets_router)


@app.get("/")
async def index():
    return FileResponse("test.html", media_type="text/html")
