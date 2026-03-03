import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as main_module
from app.errors import EMPTY_RESULT, SAMError
from app.ocr_service import OCRDetection
from app.schemas import CentroidPoint, MaskRLE, MaskResult, PointResult


def _make_image_base64() -> str:
    image = Image.new("RGB", (8, 8), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class FakeSAMService:
    def load(self) -> None:
        return

    def predict(self, image, prompt, *, multimask_output=True, max_masks=3):
        if prompt.text == "zzzzqvbnm no such object":
            raise SAMError(EMPTY_RESULT, "Model returned no masks")

        mask = MaskResult(
            id="0",
            confidence=0.91,
            mask_rle=MaskRLE(counts="abc", size=[image.height, image.width]),
        )
        point = PointResult(
            id="0",
            confidence=0.91,
            point=CentroidPoint(x=4.0, y=4.0),
        )
        return [mask][:max_masks], [point][:max_masks]


class FakeOCRService:
    def load(self) -> None:
        return

    def detect_text(self, _image):
        return [
            OCRDetection(text="red coffee mug", confidence=0.97, x1=1, y1=1, x2=6, y2=6),
            OCRDetection(text="shoe", confidence=0.91, x1=1, y1=1, x2=6, y2=6),
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "SAMService", FakeSAMService)
    monkeypatch.setattr(main_module, "OCRService", FakeOCRService)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_text_points_success_returns_single_point(client: TestClient) -> None:
    payload = {"image": _make_image_base64(), "text": "red coffee mug"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 1
    assert body["points"][0]["point"] == {"x": 4.0, "y": 4.0}
    assert body["meta"]["prompt_type"] == "text"
    assert body["meta"]["multimask_output"] is False
    assert "masks" not in body


def test_text_points_rejects_empty_text(client: TestClient) -> None:
    payload = {"image": _make_image_base64(), "text": "   "}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 422


def test_text_points_invalid_image(client: TestClient) -> None:
    payload = {"image": "not-base64", "text": "shoe"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_text_points_empty_result(client: TestClient) -> None:
    client.app.state.ocr_service.detect_text = lambda _image: []
    payload = {"image": _make_image_base64(), "text": "zzzzqvbnm no such object"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_RESULT"


def test_generic_segment_endpoint_still_supports_box_masks(client: TestClient) -> None:
    payload = {
        "image": _make_image_base64(),
        "prompt": {"box": {"x1": 1, "y1": 1, "x2": 6, "y2": 6}},
        "multimask_output": True,
        "max_masks": 1,
        "output": "masks",
    }

    response = client.post("/v1/sam/segment", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["masks"]) == 1
    assert body["meta"]["prompt_type"] == "box"
