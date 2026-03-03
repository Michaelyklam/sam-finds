from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.errors import OCR_ERROR, SAMError

logger = logging.getLogger(__name__)


@dataclass
class OCRDetection:
    text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    polygon: list[tuple[int, int]] | None = None


class OCRService:
    def __init__(self) -> None:
        self.engine: Any = None
        self.easy_reader: Any = None
        self.backend: str | None = None
        self.backend_device: str | None = None
        self.load_error: str | None = None
        self.lang = os.getenv("OCR_LANG", "en")
        self.det_limit_side_len = int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "1920"))

    def load(self) -> None:
        if self._load_paddle_gpu():
            return
        self._load_easyocr(use_gpu=False)

    def _load_paddle_gpu(self) -> bool:
        gpu_available, gpu_reason = self._is_paddle_gpu_available()
        if not gpu_available:
            self.load_error = f"PaddleOCR GPU unavailable: {gpu_reason}"
            logger.warning("%s. Falling back to EasyOCR on CPU.", self.load_error)
            return False

        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except Exception as exc:
            self.load_error = f"PaddleOCR import failed: {exc}"
            logger.exception(self.load_error)
            return False

        try:
            self.engine = PaddleOCR(
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang=self.lang,
                text_det_limit_side_len=self.det_limit_side_len,
                device="gpu",
            )
            self.backend = "paddle"
            self.backend_device = "gpu"
            self.load_error = None
            logger.info(
                "OCR backend loaded: paddle (device=gpu, lang=%s, text_det_limit_side_len=%s)",
                self.lang,
                self.det_limit_side_len,
            )
            return True
        except Exception as exc:
            self.load_error = f"PaddleOCR initialization failed: {exc}"
            logger.exception(self.load_error)
            self.engine = None
            self.backend = None
            self.backend_device = None
            return False

    def _load_easyocr(self, *, use_gpu: bool) -> bool:
        try:
            import easyocr  # type: ignore[import-untyped]
        except Exception as exc:
            self.load_error = f"EasyOCR import failed: {exc}"
            logger.exception(self.load_error)
            return False

        langs = [l.strip() for l in self.lang.split(",") if l.strip()] or ["en"]
        try:
            self.easy_reader = easyocr.Reader(langs, gpu=use_gpu)
            self.backend = "easyocr"
            self.backend_device = "gpu" if use_gpu else "cpu"
            self.load_error = None
            logger.warning(
                "OCR backend loaded: easyocr fallback (device=%s, langs=%s)",
                self.backend_device,
                ",".join(langs),
            )
            return True
        except Exception as exc:
            self.load_error = f"EasyOCR initialization failed: {exc}"
            logger.exception(self.load_error)
            self.easy_reader = None
            self.backend = None
            self.backend_device = None
            return False

    def detect_text(self, image: Image.Image) -> list[OCRDetection]:
        if self.backend is None:
            message = self.load_error or "OCR engine is not available"
            raise SAMError(OCR_ERROR, message, status_code=500)

        if self.backend == "paddle":
            try:
                return self._detect_with_paddle(image)
            except Exception as exc:
                logger.exception("PaddleOCR inference failed: %s", exc)
                if self._load_easyocr(use_gpu=False):
                    return self._detect_with_easyocr(image)
                raise SAMError(OCR_ERROR, f"OCR inference failed: {exc}", status_code=500)

        if self.backend == "easyocr":
            return self._detect_with_easyocr(image)

        raise SAMError(OCR_ERROR, "Unknown OCR backend state", status_code=500)

    def _is_paddle_gpu_available(self) -> tuple[bool, str]:
        try:
            import paddle  # type: ignore[import-untyped]
        except Exception as exc:
            return False, f"paddle import failed: {exc}"

        try:
            if not paddle.device.is_compiled_with_cuda():
                return False, "paddle build has no CUDA support"
            gpu_count = int(paddle.device.cuda.device_count())
            if gpu_count < 1:
                return False, "no CUDA device detected"
            return True, f"cuda devices={gpu_count}"
        except Exception as exc:
            return False, f"cuda capability check failed: {exc}"

    def _detect_with_paddle(self, image: Image.Image) -> list[OCRDetection]:
        assert self.engine is not None
        raw = self.engine.predict(np.array(image))
        return self._parse_paddle_output(raw)

    def _detect_with_easyocr(self, image: Image.Image) -> list[OCRDetection]:
        if self.easy_reader is None:
            raise SAMError(OCR_ERROR, "EasyOCR reader is not initialized", status_code=500)

        raw = self.easy_reader.readtext(np.array(image), detail=1)
        detections: list[OCRDetection] = []
        for item in raw:
            if not item or len(item) < 3:
                continue
            points, text, confidence = item[0], str(item[1]).strip(), float(item[2])
            if not text:
                continue
            parsed = self._detection_from_points(points, text, confidence)
            if parsed is not None:
                detections.append(parsed)
        return detections

    def _parse_paddle_output(self, raw: Any) -> list[OCRDetection]:
        detections: list[OCRDetection] = []
        if raw is None:
            return detections

        # PaddleOCR v3 can return a list of result dicts.
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            for item in raw:
                texts = item.get("rec_texts") or []
                scores = item.get("rec_scores") or []
                polys = item.get("dt_polys") or item.get("rec_polys") or []
                for text, confidence, points in zip(texts, scores, polys):
                    txt = str(text).strip()
                    if not txt:
                        continue
                    parsed = self._detection_from_points(points, txt, float(confidence))
                    if parsed is not None:
                        detections.append(parsed)
            return detections

        # Old format compatibility: [ [points, (text, score)], ... ].
        lines = raw[0] if isinstance(raw, list) and raw else []
        for line in lines:
            if not line or len(line) < 2:
                continue
            points, text_info = line[0], line[1]
            if not text_info or len(text_info) < 2:
                continue
            text = str(text_info[0]).strip()
            confidence = float(text_info[1])
            if not text:
                continue
            parsed = self._detection_from_points(points, text, confidence)
            if parsed is not None:
                detections.append(parsed)
        return detections

    def _detection_from_points(self, points: Any, text: str, confidence: float) -> OCRDetection | None:
        try:
            polygon = [
                (int(round(float(p[0]))), int(round(float(p[1]))))
                for p in points
            ]
        except Exception:
            return None
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        if not xs or not ys:
            return None

        return OCRDetection(
            text=text,
            confidence=confidence,
            x1=min(xs),
            y1=min(ys),
            x2=max(xs),
            y2=max(ys),
            polygon=polygon,
        )
