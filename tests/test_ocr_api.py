import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import ocr_main as ocr_main_module
from app.ocr_service import OCRDetection


def _make_image_base64() -> str:
    image = Image.new("RGB", (10, 8), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class FakeOCRService:
    def load(self) -> None:
        return

    def describe_backend(self):
        return {
            "backend": "paddle",
            "backend_device": "gpu",
            "lang": "en",
            "det_limit_side_len": 1920,
        }

    def detect_text(self, _image):
        return [
            OCRDetection(
                text="View Leaderboard",
                confidence=0.96,
                x1=1,
                y1=2,
                x2=8,
                y2=6,
                polygon=[(1, 2), (8, 2), (8, 6), (1, 6)],
            )
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ocr_main_module, "OCRService", FakeOCRService)
    with TestClient(ocr_main_module.app) as test_client:
        yield test_client


def test_ocr_backend_returns_runtime_details(client: TestClient) -> None:
    response = client.get("/v1/ocr/backend")

    assert response.status_code == 200
    assert response.json() == {
        "backend": "paddle",
        "backend_device": "gpu",
        "lang": "en",
        "det_limit_side_len": 1920,
    }


def test_ocr_page_returns_full_page_text_and_boxes(client: TestClient) -> None:
    response = client.post(
        "/v1/ocr/page",
        json={"image": _make_image_base64(), "include_polygons": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "View Leaderboard"
    assert body["meta"]["backend"] == "paddle"
    assert body["meta"]["num_detections"] == 1
    assert body["detections"][0]["bbox"] == {"x1": 1, "y1": 2, "x2": 8, "y2": 6}
    assert body["detections"][0]["polygon"] == [
        {"x": 1, "y": 2},
        {"x": 8, "y": 2},
        {"x": 8, "y": 6},
        {"x": 1, "y": 6},
    ]


def test_ocr_page_can_skip_polygons(client: TestClient) -> None:
    response = client.post(
        "/v1/ocr/page",
        json={"image": _make_image_base64(), "include_polygons": False},
    )

    assert response.status_code == 200
    assert response.json()["detections"][0]["polygon"] is None
