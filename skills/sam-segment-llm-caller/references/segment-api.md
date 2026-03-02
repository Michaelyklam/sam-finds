# Segment API Reference

## Endpoint
- Method: `POST`
- Path: `/v1/sam/segment`
- Content-Type: `application/json`

## Request Schema
```json
{
  "image": "<base64 PNG or JPEG>",
  "prompt": {
    "text": "cat"
  },
  "multimask_output": true,
  "max_masks": 3,
  "output": "masks"
}
```

Rules:
- `prompt` must contain exactly one of `text`, `box`, or `points`.
- `output` is `masks` or `points`.
- Defaults from API: `multimask_output=true`, `max_masks=3`, `output="masks"`.

## Prompt Variants
Text:
```json
{"prompt": {"text": "dog"}}
```

Box:
```json
{"prompt": {"box": {"x1": 100, "y1": 120, "x2": 460, "y2": 380}}}
```

Points:
```json
{"prompt": {"points": [{"x": 520, "y": 375, "label": 1}]}}
```

## Response Shapes
Masks mode:
```json
{
  "masks": [
    {
      "id": "0",
      "confidence": 0.97,
      "mask_rle": {"counts": "...", "size": [1602, 1200]}
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

Points mode:
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

## Error Codes
- `INVALID_IMAGE` (400)
- `INVALID_PROMPT` (400)
- `EMPTY_RESULT` (400)
- `MODEL_ERROR` (500)

## Code Snippets
### curl
```bash
BASE64=$(base64 -w0 image.jpg)
curl -s http://localhost:8000/v1/sam/segment \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$BASE64\",\"prompt\":{\"text\":\"dog\"},\"multimask_output\":false,\"max_masks\":2,\"output\":\"masks\"}"
```

### JavaScript (fetch)
```js
const payload = {
  image: base64Image,
  prompt: { text: "dog" },
  multimask_output: false,
  max_masks: 2,
  output: "masks",
};

const res = await fetch("http://localhost:8000/v1/sam/segment", {
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
    "prompt": {"text": "dog"},
    "multimask_output": False,
    "max_masks": 2,
    "output": "masks",
}

resp = requests.post("http://localhost:8000/v1/sam/segment", json=payload, timeout=30)
resp.raise_for_status()
data = resp.json()
```
