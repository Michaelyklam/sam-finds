# Segment API Reference

## LLM Endpoint (Use This)

- Method: `POST`
- Path: `/v1/sam/segment/text-points`
- Content-Type: `application/json`

Local network URL (for other devices on same LAN):
- `http://192.168.0.10:8000/v1/sam/segment/text-points`
- Example: `http://192.168.0.10:8000/v1/sam/segment/text-points`
- Shared base URL for all clients: `http://192.168.0.10:8000`

Request:
```json
{
  "image": "<base64 PNG or JPEG>",
  "text": "gray roll of duct tape"
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
- Prefer object-attribute phrases and avoid relational/spatial wording (`near`, `left of`, `on top of`).

## Errors
- `INVALID_IMAGE` (400)
- `EMPTY_RESULT` (400)
- `MODEL_ERROR` (500)

## Request Snippets

### curl
```bash
BASE64=$(base64 -w0 image.jpg)
curl -s http://192.168.0.10:8000/v1/sam/segment/text-points \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$BASE64\",\"text\":\"gray roll of duct tape\"}"
```

### JavaScript (fetch)
```js
const payload = {
  image: base64Image,
  text: "black camera",
};

const res = await fetch("http://192.168.0.10:8000/v1/sam/segment/text-points", {
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
    "text": "white computer mouse",
}

resp = requests.post(
    "http://192.168.0.10:8000/v1/sam/segment/text-points",
    json=payload,
    timeout=30,
)
resp.raise_for_status()
data = resp.json()
```
