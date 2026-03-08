from __future__ import annotations

import json
from pathlib import Path

from app.ocr_main import app


def test_ocr_openapi_includes_metadata_and_routes() -> None:
    schema = app.openapi()

    assert schema["info"]["title"] == "Paddle OCR Service"
    assert schema["info"]["version"] == "1.0.0"
    assert schema["info"]["summary"] == "Standalone OCR API used by sam-finds and external applications."
    assert any(tag["name"] == "ocr" for tag in schema["tags"])

    backend_operation = schema["paths"]["/v1/ocr/backend"]["get"]
    assert backend_operation["tags"] == ["ocr"]
    assert backend_operation["summary"] == "Get OCR backend details"
    assert backend_operation["operationId"] == "getOcrBackend"
    assert "500" in backend_operation["responses"]

    page_operation = schema["paths"]["/v1/ocr/page"]["post"]
    assert page_operation["tags"] == ["ocr"]
    assert page_operation["summary"] == "Run full-page OCR"
    assert page_operation["operationId"] == "postOcrPage"
    assert {"400", "422", "500"}.issubset(page_operation["responses"])


def test_ocr_openapi_includes_schema_descriptions_and_examples() -> None:
    schema = app.openapi()
    components = schema["components"]["schemas"]

    request_schema = components["OCRPageRequest"]
    assert "example" in request_schema
    assert request_schema["properties"]["image"]["description"]
    assert request_schema["properties"]["include_polygons"]["description"]

    response_schema = components["OCRPageResponse"]
    assert "example" in response_schema
    assert response_schema["properties"]["text"]["description"]
    assert response_schema["properties"]["detections"]["description"]

    error_schema = components["ErrorResponse"]
    assert "example" in error_schema


def test_committed_ocr_openapi_snapshot_matches_runtime_schema() -> None:
    runtime_schema = app.openapi()
    snapshot_path = Path(__file__).resolve().parents[1] / "docs" / "openapi" / "ocr-service.openapi.json"
    snapshot_schema = json.loads(snapshot_path.read_text())

    assert snapshot_schema == runtime_schema
