from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.errors import SAMError, sam_error_handler
from app.routes.segment import router as segment_router
from app.sam_service import SAMService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service = SAMService()
    service.load()
    app.state.sam_service = service
    yield


app = FastAPI(title="SAM3 Segmentation API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_exception_handler(SAMError, sam_error_handler)  # type: ignore[arg-type]
app.include_router(segment_router)


@app.get("/")
async def index():
    return FileResponse("test.html", media_type="text/html")
