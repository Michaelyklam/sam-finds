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
    payload = {"image": _make_image_base64(), "text": "red coffee mug", "text_mode": "visual"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 1
    assert body["points"][0]["point"] == {"x": 4.0, "y": 4.0}
    assert body["meta"]["prompt_type"] == "text"
    assert body["meta"]["multimask_output"] is False
    assert "masks" not in body


def test_text_points_default_visual_mode_does_not_use_ocr(client: TestClient) -> None:
    def _should_not_run(_image):
        raise AssertionError("OCR should not run in default visual mode")

    client.app.state.ocr_service.detect_text = _should_not_run
    payload = {"image": _make_image_base64(), "text": "red coffee mug", "text_mode": "visual"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["model"] == "sam3"


def test_text_points_screen_text_mode_uses_ocr(client: TestClient) -> None:
    payload = {"image": _make_image_base64(), "text": "red coffee mug", "text_mode": "screen_text"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["model"] == "sam3+ocr"


def test_text_points_screen_text_mode_does_not_require_sam_refine(client: TestClient) -> None:
    def _sam_should_not_run(*_args, **_kwargs):
        raise AssertionError("SAM refine should not run for OCR hits in screen_text mode")

    client.app.state.sam_service.predict = _sam_should_not_run
    payload = {"image": _make_image_base64(), "text": "red coffee mug", "text_mode": "screen_text"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 1
    assert body["meta"]["model"] == "sam3+ocr"


def test_text_points_screen_text_mode_uses_ocr_polygon_centroid(client: TestClient) -> None:
    client.app.state.ocr_service.detect_text = lambda _image: [
        OCRDetection(
            text="triangle label",
            confidence=0.95,
            x1=0,
            y1=0,
            x2=8,
            y2=8,
            polygon=[(0, 0), (8, 0), (0, 8)],
        )
    ]
    client.app.state.sam_service.predict = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("SAM refine should not run for OCR hits in screen_text mode")
    )

    payload = {"image": _make_image_base64(), "text": "triangle label", "text_mode": "screen_text"}
    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 200
    body = response.json()
    point = body["points"][0]["point"]
    assert point["x"] == pytest.approx(2.67, abs=0.01)
    assert point["y"] == pytest.approx(2.67, abs=0.01)


def test_text_points_rejects_empty_text(client: TestClient) -> None:
    payload = {"image": _make_image_base64(), "text": "   ", "text_mode": "visual"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 422


def test_text_points_invalid_image(client: TestClient) -> None:
    payload = {"image": "not-base64", "text": "shoe", "text_mode": "visual"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_text_points_empty_result(client: TestClient) -> None:
    client.app.state.ocr_service.detect_text = lambda _image: []
    payload = {
        "image": _make_image_base64(),
        "text": "zzzzqvbnm no such object",
        "text_mode": "visual",
    }

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_RESULT"


def test_generic_segment_endpoint_still_supports_box_masks(client: TestClient) -> None:
    payload = {
        "image": _make_image_base64(),
        "prompt": {"box": {"x1": 1, "y1": 1, "x2": 6, "y2": 6}},
        "text_mode": "visual",
        "multimask_output": True,
        "max_masks": 1,
        "output": "masks",
    }

    response = client.post("/v1/sam/segment", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["masks"]) == 1
    assert body["meta"]["prompt_type"] == "box"


def test_text_points_requires_text_mode_with_hint(client: TestClient) -> None:
    payload = {"image": _make_image_base64(), "text": "red coffee mug"}

    response = client.post("/v1/sam/segment/text-points", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert "text_mode" in body["error"]["hint"]
    assert "visual" in body["error"]["hint"]
