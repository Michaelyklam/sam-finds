from __future__ import annotations

import sys
import types

from PIL import Image

from app.ocr_service import OCRService


class _FakeCuda:
    def __init__(self, count: int):
        self._count = count

    def device_count(self) -> int:
        return self._count


class _FakePaddleDevice:
    def __init__(self, *, compiled_with_cuda: bool, device_count: int):
        self._compiled_with_cuda = compiled_with_cuda
        self.cuda = _FakeCuda(device_count)

    def is_compiled_with_cuda(self) -> bool:
        return self._compiled_with_cuda


class _FakePaddleOCR:
    init_kwargs: dict | None = None
    init_calls: int = 0
    raise_on_predict: bool = False

    def __init__(self, **kwargs):
        _FakePaddleOCR.init_kwargs = kwargs
        _FakePaddleOCR.init_calls += 1

    def predict(self, _image):
        if _FakePaddleOCR.raise_on_predict:
            raise RuntimeError("forced paddle failure")
        return [
            {
                "rec_texts": ["Okay"],
                "rec_scores": [0.91],
                "dt_polys": [[[1, 1], [20, 1], [20, 20], [1, 20]]],
            }
        ]


class _FakeEasyReader:
    init_gpu: bool | None = None

    def __init__(self, _langs, gpu: bool = False):
        _FakeEasyReader.init_gpu = gpu

    def readtext(self, _image, detail=1):
        assert detail == 1
        return [([[2, 2], [30, 2], [30, 30], [2, 30]], "Settings", 0.96)]


def _install_fake_modules(monkeypatch, *, paddle_cuda: bool, paddle_device_count: int) -> None:
    _FakePaddleOCR.init_kwargs = None
    _FakePaddleOCR.init_calls = 0
    _FakePaddleOCR.raise_on_predict = False
    _FakeEasyReader.init_gpu = None

    fake_paddle = types.SimpleNamespace(
        device=_FakePaddleDevice(
            compiled_with_cuda=paddle_cuda,
            device_count=paddle_device_count,
        )
    )
    fake_paddleocr = types.SimpleNamespace(PaddleOCR=_FakePaddleOCR)
    fake_easyocr = types.SimpleNamespace(Reader=_FakeEasyReader)

    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)


def test_prefers_paddle_gpu_when_available(monkeypatch) -> None:
    _install_fake_modules(monkeypatch, paddle_cuda=True, paddle_device_count=1)

    service = OCRService()
    service.load()

    assert service.backend == "paddle"
    assert service.backend_device == "gpu"
    assert _FakePaddleOCR.init_calls == 1
    assert _FakePaddleOCR.init_kwargs is not None
    assert _FakePaddleOCR.init_kwargs["device"] == "gpu"


def test_falls_back_to_easyocr_cpu_when_paddle_gpu_unavailable(monkeypatch) -> None:
    _install_fake_modules(monkeypatch, paddle_cuda=False, paddle_device_count=0)

    service = OCRService()
    service.load()

    assert service.backend == "easyocr"
    assert service.backend_device == "cpu"
    assert _FakePaddleOCR.init_calls == 0
    assert _FakeEasyReader.init_gpu is False


def test_falls_back_to_easyocr_cpu_when_paddle_inference_fails(monkeypatch) -> None:
    _install_fake_modules(monkeypatch, paddle_cuda=True, paddle_device_count=1)
    _FakePaddleOCR.raise_on_predict = True

    service = OCRService()
    service.load()
    detections = service.detect_text(Image.new("RGB", (64, 64), "white"))

    assert service.backend == "easyocr"
    assert service.backend_device == "cpu"
    assert len(detections) == 1
    assert detections[0].text == "Settings"
