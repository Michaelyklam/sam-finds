# Segment API Reference

## Base URL

- `http://192.168.0.10:8000`

## Endpoints

### Named controls (recommended for button text)
- Method: `POST`
- Path: `/v1/ui/click-targets`
- Content-Type: `application/json`

Request:
```json
{
  "image": "<base64 PNG or JPEG>",
  "target_text": "View Leaderboard button",
  "max_candidates": 3,
  "use_sam_refine": true
}
```

Response:
```json
{
  "candidates": [
    {
      "id": "0",
      "point": {"x": 485.99, "y": 1171.94},
      "score": 0.91,
      "ocr_text": "View Leaderboard",
      "ocr_confidence": 0.96,
      "sam_confidence": 0.78,
      "bbox": {"x1": 412, "y1": 1110, "x2": 560, "y2": 1234}
    }
  ],
  "meta": {
    "pipeline": "ocr+sam",
    "target_text": "View Leaderboard button",
    "num_ocr_hits": 2,
    "num_candidates": 1,
    "latency_ms": 842
  }
}
```

### Same interaction format as tester
- Method: `POST`
- Path: `/v1/ui/segment`
- Content-Type: `application/json`

Request/response schema matches `/v1/sam/segment` (`prompt`, `output`, `max_masks`), but text prompts are OCR-assisted.

Text prompt mode:
- `text_mode: "screen_text"` enables OCR assist and returns OCR-native points/masks.
- `text_mode: "visual"` bypasses OCR and uses plain SAM text prompting.
- `text_mode` is required.

### LLM-friendly text to points
- Method: `POST`
- Path: `/v1/sam/segment/text-points`
- Content-Type: `application/json`

Request:
```json
{
  "image": "<base64 PNG or JPEG>",
  "text": "white computer mouse",
  "text_mode": "visual"
}
```

Response:
```json
{
  "points": [
    {
      "id": "0",
      "confidence": 0.97,
      "point": {"x": 412.35, "y": 301.78}
    }
  ],
  "meta": {
    "image_width": 1200,
    "image_height": 1602,
    "model": "sam3",
    "prompt_type": "text",
    "multimask_output": false
  }
}
```

## Errors
- `INVALID_REQUEST` (422)
- `INVALID_IMAGE` (400)
- `EMPTY_RESULT` (400)
- `MODEL_ERROR` (500)
- `TEXT_NOT_FOUND` (400)
- `OCR_ERROR` (500)

## Request Snippets

### curl
```bash
BASE64=$(base64 -w0 image.jpg)
curl -s http://192.168.0.10:8000/v1/ui/click-targets \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$BASE64\",\"target_text\":\"View Leaderboard button\",\"max_candidates\":3,\"use_sam_refine\":true}"
```

### JavaScript (fetch)
```js
const payload = {
  image: base64Image,
  target_text: "View Leaderboard button",
  max_candidates: 3,
  use_sam_refine: true,
};

const res = await fetch("http://192.168.0.10:8000/v1/ui/click-targets", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

if (!res.ok) throw new Error(`SAM request failed: ${res.status}`);
const data = await res.json();
```

### Python (requests)
```python
import requests

payload = {
    "image": base64_image,
    "target_text": "View Leaderboard button",
    "max_candidates": 3,
    "use_sam_refine": True,
}

resp = requests.post(
    "http://192.168.0.10:8000/v1/ui/click-targets",
    json=payload,
    timeout=30,
)
resp.raise_for_status()
data = resp.json()
```
