from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from app.errors import OCR_ERROR, SAMError
from app.ocr_remote_service import RemoteOCRService


def _client_factory(httpx_client_cls, handler):
    transport = httpx.MockTransport(handler)

    def _factory(*args, **kwargs):
        return httpx_client_cls(
            transport=transport,
            base_url=kwargs.get("base_url"),
            timeout=kwargs.get("timeout"),
        )

    return _factory


def _request_json(request: httpx.Request) -> dict:
    return json.loads(request.content.decode("utf-8"))


def test_remote_ocr_service_loads_backend_and_maps_detections(monkeypatch) -> None:
    httpx_client_cls = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/ocr/backend":
            return httpx.Response(
                200,
                json={
                    "backend": "paddle",
                    "backend_device": "gpu",
                    "lang": "en",
                    "det_limit_side_len": 1920,
                },
            )
        if request.method == "POST" and request.url.path == "/v1/ocr/page":
            payload = _request_json(request)
            image = Image.open(io.BytesIO(base64.b64decode(payload["image"]))).convert("RGB")
            assert image.size == (6, 4)
            return httpx.Response(
                200,
                json={
                    "text": "Settings",
                    "detections": [
                        {
                            "id": "0",
                            "text": "Settings",
                            "confidence": 0.94,
                            "bbox": {"x1": 1, "y1": 1, "x2": 5, "y2": 3},
                            "polygon": [
                                {"x": 1, "y": 1},
                                {"x": 5, "y": 1},
                                {"x": 5, "y": 3},
                                {"x": 1, "y": 3},
                            ],
                        }
                    ],
                    "meta": {
                        "image_width": 6,
                        "image_height": 4,
                        "backend": "paddle",
                        "backend_device": "gpu",
                        "lang": "en",
                        "num_detections": 1,
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    monkeypatch.setattr(httpx, "Client", _client_factory(httpx_client_cls, handler))

    service = RemoteOCRService("http://ocr-service:8001")
    service.load()

    detections = service.detect_text(Image.new("RGB", (6, 4), "white"))
    service.close()

    assert service.describe_backend()["backend"] == "paddle"
    assert len(detections) == 1
    assert detections[0].text == "Settings"
    assert detections[0].polygon == [(1, 1), (5, 1), (5, 3), (1, 3)]


def test_remote_ocr_service_raises_ocr_error_for_http_failures(monkeypatch) -> None:
    httpx_client_cls = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/ocr/backend":
            return httpx.Response(
                200,
                json={
                    "backend": "paddle",
                    "backend_device": "gpu",
                    "lang": "en",
                    "det_limit_side_len": 1920,
                },
            )
        return httpx.Response(503, json={"error": {"message": "OCR backend unavailable"}})

    monkeypatch.setattr(httpx, "Client", _client_factory(httpx_client_cls, handler))

    service = RemoteOCRService("http://ocr-service:8001")
    service.load()

    with pytest.raises(SAMError) as exc_info:
        service.detect_text(Image.new("RGB", (6, 4), "white"))

    service.close()

    assert exc_info.value.code == OCR_ERROR
    assert "unavailable" in exc_info.value.message.lower()
