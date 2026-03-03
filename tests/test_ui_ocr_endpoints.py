import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as main_module
from app.errors import EMPTY_RESULT
from app.ocr_service import OCRDetection
from app.schemas import CentroidPoint, MaskRLE, MaskResult, PointResult


def _make_image_base64() -> str:
    image = Image.new("RGB", (1200, 1602), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class FakeSAMService:
    def load(self) -> None:
        return

    def predict(self, image, prompt, *, multimask_output=True, max_masks=3):
        if prompt.box is not None:
            b = prompt.box
            cx = round((b.x1 + b.x2) / 2.0, 2)
            cy = round((b.y1 + b.y2) / 2.0, 2)
            score = 0.82
            mask = MaskResult(
                id="0",
                confidence=score,
                mask_rle=MaskRLE(counts="abc", size=[image.height, image.width]),
            )
            point = PointResult(
                id="0",
                confidence=score,
                point=CentroidPoint(x=cx, y=cy),
            )
            return [mask][:max_masks], [point][:max_masks]

        if prompt.text is not None and prompt.text.strip().lower() == "no object":
            from app.errors import SAMError

            raise SAMError(EMPTY_RESULT, "Model returned no masks")

        score = 0.6
        mask = MaskResult(
            id="0",
            confidence=score,
            mask_rle=MaskRLE(counts="xyz", size=[image.height, image.width]),
        )
        point = PointResult(
            id="0",
            confidence=score,
            point=CentroidPoint(x=600.0, y=801.0),
        )
        return [mask][:max_masks], [point][:max_masks]


class FakeOCRService:
    def load(self) -> None:
        return

    def detect_text(self, _image):
        return [
            OCRDetection(text="View Leaderboard", confidence=0.98, x1=380, y1=1100, x2=600, y2=1240),
            OCRDetection(text="View Leaderboard", confidence=0.9, x1=620, y1=1100, x2=840, y2=1240),
            OCRDetection(text="Settings", confidence=0.95, x1=90, y1=200, x2=260, y2=300),
            OCRDetection(text="Okay", confidence=0.93, x1=420, y1=1300, x2=570, y2=1390),
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "SAMService", FakeSAMService)
    monkeypatch.setattr(main_module, "OCRService", FakeOCRService)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_click_targets_returns_ranked_candidates(client: TestClient) -> None:
    payload = {
        "image": _make_image_base64(),
        "target_text": "View Leaderboard button",
        "max_candidates": 2,
        "use_sam_refine": True,
    }

    response = client.post("/v1/ui/click-targets", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["pipeline"] == "ocr+sam"
    assert body["meta"]["num_candidates"] == 2
    assert len(body["candidates"]) == 2
    assert body["candidates"][0]["score"] >= body["candidates"][1]["score"]
    assert body["candidates"][0]["ocr_text"] == "View Leaderboard"


def test_click_targets_text_not_found(client: TestClient) -> None:
    payload = {
        "image": _make_image_base64(),
        "target_text": "zzzzqvbnm no such control",
    }

    response = client.post("/v1/ui/click-targets", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "TEXT_NOT_FOUND"


def test_ui_segment_points_text_uses_ocr_assist(client: TestClient) -> None:
    payload = {
        "image": _make_image_base64(),
        "prompt": {"text": "View Leaderboard button"},
        "text_mode": "screen_text",
        "multimask_output": True,
        "max_masks": 2,
        "output": "points",
    }

    response = client.post("/v1/ui/segment", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 2
    assert body["meta"]["model"] == "sam3+ocr"
    assert body["meta"]["prompt_type"] == "text"
    assert body["meta"]["multimask_output"] is False


def test_ui_segment_masks_text_uses_ocr_assist(client: TestClient) -> None:
    payload = {
        "image": _make_image_base64(),
        "prompt": {"text": "Okay"},
        "text_mode": "screen_text",
        "multimask_output": True,
        "max_masks": 1,
        "output": "masks",
    }

    response = client.post("/v1/ui/segment", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["masks"]) == 1
    assert body["meta"]["model"] == "sam3+ocr"


def test_ui_segment_screen_text_mode_does_not_require_sam_refine(client: TestClient) -> None:
    def _sam_should_not_run(*_args, **_kwargs):
        raise AssertionError("SAM refine should not run for OCR hits in screen_text mode")

    client.app.state.sam_service.predict = _sam_should_not_run
    payload = {
        "image": _make_image_base64(),
        "prompt": {"text": "Okay"},
        "text_mode": "screen_text",
        "multimask_output": True,
        "max_masks": 1,
        "output": "points",
    }

    response = client.post("/v1/ui/segment", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 1
    assert body["meta"]["model"] == "sam3+ocr"


def test_ui_segment_text_falls_back_to_plain_sam_when_no_ocr_match(client: TestClient) -> None:
    payload = {
        "image": _make_image_base64(),
        "prompt": {"text": "cat"},
        "text_mode": "screen_text",
        "multimask_output": True,
        "max_masks": 1,
        "output": "points",
    }

    response = client.post("/v1/ui/segment", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 1
    assert body["meta"]["model"] == "sam3"


def test_ui_segment_visual_mode_bypasses_ocr(client: TestClient) -> None:
    def _should_not_run(_image):
        raise AssertionError("OCR should not run when text_mode=visual")

    client.app.state.ocr_service.detect_text = _should_not_run
    payload = {
        "image": _make_image_base64(),
        "prompt": {"text": "View Leaderboard button"},
        "text_mode": "visual",
        "multimask_output": True,
        "max_masks": 1,
        "output": "points",
    }

    response = client.post("/v1/ui/segment", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 1
    assert body["meta"]["model"] == "sam3"


def test_ui_segment_requires_text_mode_with_hint(client: TestClient) -> None:
    payload = {
        "image": _make_image_base64(),
        "prompt": {"text": "View Leaderboard button"},
        "multimask_output": True,
        "max_masks": 1,
        "output": "points",
    }

    response = client.post("/v1/ui/segment", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert "text_mode" in body["error"]["hint"]
    assert "screen_text" in body["error"]["hint"]
