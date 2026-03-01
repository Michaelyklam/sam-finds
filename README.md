# SAM Finds

A FastAPI service wrapping Meta's [Segment Anything Model 3](https://github.com/facebookresearch/sam3). Accepts an image + prompt and returns ranked segmentation masks with COCO RLE encoding and confidence scores.

## Quick Start

```bash
docker compose up --build -d
# Model loads on first startup (~30s). Check readiness:
docker compose logs -f
```

The API is available at `http://localhost:8000`. A built-in mask/point viewer is served at `/`.

### Network Access

The API listens on all interfaces (`0.0.0.0:8000`), so any device on the same local network can reach it. Find the host machine's IP:

```bash
hostname -I | awk '{print $1}'
```

Then from another device:

- **Viewer**: `http://<HOST_IP>:8000/`
- **API**: `http://<HOST_IP>:8000/v1/sam/segment`

Example — segment an image from a phone/laptop on the same Wi-Fi:

```bash
# Encode an image and call the API
BASE64=$(base64 -w0 photo.jpg)

# Get masks (default)
curl -s http://192.168.0.10:8000/v1/sam/segment \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$BASE64\",\"prompt\":{\"text\":\"dog\"}}" | jq .

# Get centroid points instead
curl -s http://192.168.0.10:8000/v1/sam/segment \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$BASE64\",\"prompt\":{\"text\":\"dog\"},\"output\":\"points\"}" | jq .
```

### Requirements

- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Docker with Compose v2

## API

### `POST /v1/sam/segment`

Segment an image using a text, box, or point prompt.

#### Request

```json
{
  "image": "<base64-encoded PNG or JPEG>",
  "prompt": { ... },
  "multimask_output": true,
  "max_masks": 3,
  "output": "masks"
}
```

The `output` field controls the response format:

| Value | Default | Description |
|-------|---------|-------------|
| `"masks"` | Yes | Return full COCO RLE masks with confidence scores |
| `"points"` | No | Return centroid points (center of each mask) with confidence scores |

The `prompt` field must contain **exactly one** of:

| Type | Field | Format | Example |
|------|-------|--------|---------|
| Text | `text` | Natural language description | `{"text": "cat"}` |
| Box | `box` | `{x1, y1, x2, y2}` in pixels | `{"box": {"x1": 100, "y1": 100, "x2": 400, "y2": 300}}` |
| Points | `points` | List of `{x, y, label}` (1=fg, 0=bg) | `{"points": [{"x": 520, "y": 375, "label": 1}]}` |

**Text prompts** are SAM3's primary interface and produce the best results. Box prompts give very high confidence when the box tightly fits the object. Point prompts are approximated via small bounding boxes (SAM3's processor doesn't natively support point prompts).

#### Response

```json
{
  "masks": [
    {
      "id": "0",
      "confidence": 0.97,
      "mask_rle": {
        "counts": "...",
        "size": [1602, 1200]
      }
    }
  ],
  "meta": {
    "image_width": 1200,
    "image_height": 1602,
    "model": "sam3",
    "prompt_type": "text",
    "multimask_output": true
  }
}
```

Masks are sorted by confidence descending and capped at `max_masks`. The `mask_rle` field uses [COCO RLE format](https://github.com/cocodataset/cocoapi) (column-major run-length encoding).

#### Points Response (`output: "points"`)

```json
{
  "points": [
    {
      "id": "0",
      "confidence": 0.97,
      "point": { "x": 412.35, "y": 301.78 }
    }
  ],
  "meta": { ... }
}
```

Each point is the centroid (center of mass) of the corresponding segmentation mask. Coordinates are in pixel space relative to the original image dimensions.

#### Errors

```json
{"error": {"code": "INVALID_IMAGE", "message": "Could not decode image: ..."}}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_IMAGE` | 400 | Base64 decoding or image parsing failed |
| `INVALID_PROMPT` | 400 | Prompt field missing or malformed |
| `MODEL_ERROR` | 500 | SAM3 inference failed |
| `EMPTY_RESULT` | 400 | Model returned no masks for the given prompt |

## Decoding Masks

### Python

```python
from pycocotools import mask as mask_utils
import numpy as np

rle = {"counts": "...", "size": [H, W]}
rle["counts"] = rle["counts"].encode()  # pycocotools expects bytes
binary_mask = mask_utils.decode(rle)    # numpy (H, W) of 0s and 1s
```

### JavaScript

The COCO RLE string uses a custom variable-length encoding. See `test.html` for a working decoder. Key details:

- Each character encodes 5 data bits + 1 continuation bit, offset by ASCII 48
- Sign extension (not zigzag): when the high data bit is set and no more bytes follow, upper bits are filled with 1s
- Delta encoding starts from the 4th value: `cnts[i] += cnts[i-2]` for `i > 2`
- The flat mask array is column-major: pixel at `(x, y)` is at index `x * height + y`

## Project Structure

```
app/
├── main.py           # FastAPI app, lifespan (model loading), CORS, HTML serving
├── schemas.py        # Pydantic request/response models
├── sam_service.py    # SAM3 inference wrapper (Sam3Processor)
├── errors.py         # SAMError exception + JSON error handler
└── routes/
    └── segment.py    # POST /v1/sam/segment handler
Dockerfile            # PyTorch 2.9.0 + CUDA 13.0, SAM3 from git, model pre-downloaded
docker-compose.yml    # GPU passthrough, HF cache volume
test.html             # Interactive mask viewer (served at /)
```

## Architecture Notes

- **Model loading**: SAM3 loads once at startup via FastAPI lifespan into `app.state.sam_service`. The checkpoint is pre-downloaded into the Docker image from a [public mirror](https://huggingface.co/1038lab/sam3) to avoid gated repo auth at runtime.
- **Sam3Processor**: The service uses SAM3's `Sam3Processor` API — `set_image()` → `set_text_prompt()` or `add_geometric_prompt()`. Results (masks, scores, boxes) are returned in the inference state dict.
- **Box format**: The API accepts pixel-coordinate boxes (`x1,y1,x2,y2`) and internally converts to normalized center-xy-width-height format for `add_geometric_prompt()`.
- **RLE encoding**: Binary masks from the model are encoded via `pycocotools.mask.encode()` before returning. The `counts` field is decoded from bytes to a UTF-8 string for JSON transport.
- **CORS**: Enabled for all origins so the viewer works when accessed over the network.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM3_CHECKPOINT` | `/models/sam3.pt` | Path to SAM3 `.pt` checkpoint |
| `HF_TOKEN` | (none) | HuggingFace token (only needed if downloading from gated `facebook/sam3` repo) |
