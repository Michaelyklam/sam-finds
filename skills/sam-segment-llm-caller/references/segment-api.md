# Segment API Reference

## LLM Endpoint (Use This)

- Method: `POST`
- Path: `/v1/sam/segment/text-points`
- Content-Type: `application/json`

Request:
```json
{
  "image": "<base64 PNG or JPEG>",
  "text": "red coffee mug near the keyboard"
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

Contract notes:
- Endpoint accepts text only.
- Endpoint returns points only.
- Server enforces balanced settings (`multimask_output=false`, `max_masks=1`).

## Errors
- `INVALID_IMAGE` (400)
- `EMPTY_RESULT` (400)
- `MODEL_ERROR` (500)

## Request Snippets

### curl
```bash
BASE64=$(base64 -w0 image.jpg)
curl -s http://localhost:8000/v1/sam/segment/text-points \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$BASE64\",\"text\":\"red coffee mug near the keyboard\"}"
```

### JavaScript (fetch)
```js
const payload = {
  image: base64Image,
  text: "red coffee mug near the keyboard",
};

const res = await fetch("http://localhost:8000/v1/sam/segment/text-points", {
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
    "text": "red coffee mug near the keyboard",
}

resp = requests.post(
    "http://localhost:8000/v1/sam/segment/text-points",
    json=payload,
    timeout=30,
)
resp.raise_for_status()
data = resp.json()
```
