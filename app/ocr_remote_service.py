from __future__ import annotations

from typing import Any

import httpx
from PIL import Image

from app.errors import OCR_ERROR, SAMError
from app.image_utils import encode_image_base64
from app.ocr_service import OCRDetection


class RemoteOCRService:
    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client: httpx.Client | None = None
        self.backend: str | None = None
        self.backend_device: str | None = None
        self.lang: str | None = None
        self.det_limit_side_len: int | None = None

    def load(self) -> None:
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds)
        payload = self._request("GET", "/v1/ocr/backend")
        self.backend = str(payload.get("backend") or "remote")
        self.backend_device = str(payload.get("backend_device") or "unknown")
        self.lang = str(payload.get("lang") or "en")
        self.det_limit_side_len = int(payload.get("det_limit_side_len") or 0)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def describe_backend(self) -> dict[str, str | int]:
        return {
            "backend": self.backend or "remote",
            "backend_device": self.backend_device or "unknown",
            "lang": self.lang or "en",
            "det_limit_side_len": self.det_limit_side_len or 0,
        }

    def detect_text(self, image: Image.Image) -> list[OCRDetection]:
        payload = self._request(
            "POST",
            "/v1/ocr/page",
            json={"image": encode_image_base64(image), "include_polygons": True},
        )
        detections: list[OCRDetection] = []
        for item in payload.get("detections", []):
            bbox = item.get("bbox") or {}
            polygon_items = item.get("polygon")
            polygon = None
            if isinstance(polygon_items, list):
                polygon = []
                for point in polygon_items:
                    if not isinstance(point, dict):
                        continue
                    polygon.append((int(point.get("x", 0)), int(point.get("y", 0))))
            detections.append(
                OCRDetection(
                    text=str(item.get("text", "")).strip(),
                    confidence=float(item.get("confidence", 0.0)),
                    x1=int(bbox.get("x1", 0)),
                    y1=int(bbox.get("y1", 0)),
                    x2=int(bbox.get("x2", 0)),
                    y2=int(bbox.get("y2", 0)),
                    polygon=polygon,
                )
            )
        return detections

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Remote OCR client is not initialized")

        try:
            response = self.client.request(method, path, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Remote OCR request failed: {exc}") from exc

        if response.is_success:
            data = response.json()
            if isinstance(data, dict):
                return data
            raise RuntimeError("Remote OCR returned a non-object JSON payload")

        message = f"Remote OCR returned HTTP {response.status_code}"
        try:
            data = response.json()
            error = data.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
        except Exception:
            pass
        raise SAMError(OCR_ERROR, message, status_code=500)
