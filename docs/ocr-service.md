# OCR Service

## 1. Overview

The OCR service is a standalone FastAPI application for full-page text extraction. It exposes a small API surface that other apps can call directly:

- `GET /v1/ocr/backend`
- `POST /v1/ocr/page`

`sam-finds` also uses this service internally when `OCR_SERVICE_URL` is set, so the OCR API documented here is the same one consumed by the segmentation stack.

Primary documentation surfaces:

- Swagger UI: `http://localhost:8001/docs`
- OpenAPI JSON: `http://localhost:8001/openapi.json`
- Repo README entrypoint: [README.md](../README.md)

## 2. When To Use The OCR Service Vs SAM Endpoints

Use the OCR service when:

- you need full-page text extraction from a screenshot or image
- you want structured OCR detections with text, confidence, boxes, and polygons
- your app only needs text-region discovery and not visual segmentation
- you are building another service that wants to reason about screen text directly

Use SAM endpoints when:

- you need mask or point outputs for visual objects
- you want segmentation driven by natural-language visual prompts
- you need OCR-assisted selection inside `sam-finds`, such as named UI targeting

Practical rule:

- OCR service: “find text on this page”
- SAM service: “find or click this visual thing”

## 3. Quick Start

Run the full stack:

```bash
docker compose up --build -d
docker compose logs -f ocr-service sam-finds
```

Check the OCR service runtime:

```bash
curl -s http://localhost:8001/v1/ocr/backend | jq .
```

Run a page OCR request:

```bash
BASE64=$(base64 -w0 screenshot.png)

curl -s http://localhost:8001/v1/ocr/page \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$BASE64\",\"include_polygons\":true}" | jq .
```

## 4. Standalone Deployment

### Full stack via Compose

This is the default repo flow and starts both services:

```bash
docker compose up --build -d
```

Services:

- `sam-finds`: `http://localhost:8000`
- `ocr-service`: `http://localhost:8001`

### OCR-only container

Build and run the OCR service without `sam-finds`:

```bash
docker build -f Dockerfile.ocr -t paddle-ocr-service .

docker run --rm \
  --gpus all \
  -p 8001:8001 \
  -e OCR_LANG=en \
  -e OCR_DET_LIMIT_SIDE_LEN=1920 \
  paddle-ocr-service
```

### Relevant environment variables

- `OCR_LANG`
  - default: `en`
  - controls the language set passed to the OCR backend
- `OCR_DET_LIMIT_SIDE_LEN`
  - default: `1920`
  - controls the OCR detector side-length limit before internal resizing

## 5. API Endpoints

### `GET /v1/ocr/backend`

Purpose:

- discover the currently active OCR backend
- confirm device/runtime behavior during startup
- inspect language and detector settings

Example response:

```json
{
  "backend": "paddle",
  "backend_device": "gpu",
  "lang": "en",
  "det_limit_side_len": 1920
}
```

Recommended usage:

- call once at client startup or during health checks
- log the response for debugging and support

### `POST /v1/ocr/page`

Purpose:

- run full-page OCR on a base64-encoded PNG or JPEG image
- return a convenience text aggregate plus structured OCR detections

Request body:

```json
{
  "image": "<base64-encoded PNG or JPEG>",
  "include_polygons": true
}
```

Example response:

```json
{
  "text": "View Leaderboard\nSettings",
  "detections": [
    {
      "id": "0",
      "text": "View Leaderboard",
      "confidence": 0.96,
      "bbox": {
        "x1": 412,
        "y1": 1110,
        "x2": 560,
        "y2": 1234
      },
      "polygon": [
        { "x": 412, "y": 1110 },
        { "x": 560, "y": 1110 },
        { "x": 560, "y": 1234 },
        { "x": 412, "y": 1234 }
      ]
    }
  ],
  "meta": {
    "image_width": 1200,
    "image_height": 1602,
    "backend": "paddle",
    "backend_device": "gpu",
    "lang": "en",
    "num_detections": 2
  }
}
```

## 6. Request / Response Schema Reference

### Request fields

#### `image`

- type: string
- format: base64-encoded PNG or JPEG bytes
- required: yes

Guidance:

- send the original screenshot resolution when possible
- avoid pre-cropping unless you intentionally want OCR only for a subsection

#### `include_polygons`

- type: boolean
- required: no
- default: `true`

Use `true` when:

- geometry matters
- you want polygon-aware overlays or click heuristics
- you want the richest downstream data

Use `false` when:

- you only need text and bounding boxes
- you want lighter response payloads

### Response fields

#### `text`

- convenience field containing newline-joined detected text
- useful for quick logging and debugging
- not the best source for structured automation

#### `detections`

This is the structured source of truth for integrations.

Each detection contains:

- `id`: response-local identifier
- `text`: recognized text
- `confidence`: OCR confidence score
- `bbox`: axis-aligned bounding box in pixels
- `polygon`: optional polygon geometry

#### `meta`

Contains:

- image dimensions
- active backend
- device
- language configuration
- detection count

## 7. Error Handling

The OCR service returns errors in this envelope:

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Could not decode image: Incorrect padding",
    "hint": "Send `image` as base64-encoded PNG/JPEG bytes."
  }
}
```

Common error codes:

### `INVALID_IMAGE`

Meaning:

- the provided `image` field could not be decoded as base64 PNG/JPEG bytes

Typical fix:

- verify the base64 encoding step
- ensure the payload does not include extra URI prefixes unless your client strips them first

### `INVALID_REQUEST`

Meaning:

- the request body failed schema validation

Typical fix:

- validate required fields
- ensure JSON shape matches the OpenAPI contract

### `OCR_ERROR`

Meaning:

- the OCR backend failed to initialize or failed during inference

Typical fix:

- inspect `GET /v1/ocr/backend`
- check container logs
- verify GPU/runtime availability if PaddleOCR is expected

## 8. Integration Examples

### cURL

```bash
BASE64=$(base64 -w0 screenshot.png)

curl -s http://localhost:8001/v1/ocr/page \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$BASE64\",\"include_polygons\":true}" | jq .
```

### JavaScript (`fetch`)

```js
import { readFile } from "node:fs/promises";

const imageBuffer = await readFile("screenshot.png");
const imageBase64 = imageBuffer.toString("base64");

const response = await fetch("http://localhost:8001/v1/ocr/page", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    image: imageBase64,
    include_polygons: true,
  }),
});

if (!response.ok) {
  const error = await response.json();
  throw new Error(JSON.stringify(error, null, 2));
}

const result = await response.json();
console.log(result.meta.backend, result.detections.length);
```

### Python (`requests`)

```python
import base64
from pathlib import Path

import requests

image_base64 = base64.b64encode(Path("screenshot.png").read_bytes()).decode("utf-8")

response = requests.post(
    "http://localhost:8001/v1/ocr/page",
    json={
        "image": image_base64,
        "include_polygons": True,
    },
    timeout=30,
)
response.raise_for_status()

result = response.json()
print(result["meta"]["backend"], len(result["detections"]))
```

### `sam-finds` integration

`sam-finds` consumes the OCR service via:

```bash
OCR_SERVICE_URL=http://ocr-service:8001
```

That means external consumers and `sam-finds` share the same OCR contract.

## 9. Operational Notes

- The OCR service prefers PaddleOCR on GPU when CUDA is available.
- If PaddleOCR GPU is unavailable or fails, the service falls back to EasyOCR on CPU.
- `GET /v1/ocr/backend` is the fastest way to confirm active runtime behavior.
- For automation workflows, use `detections` rather than the concatenated `text` field.
- `bbox` is axis-aligned even when `polygon` is non-rectangular.

Payload guidance:

- use `include_polygons=true` for richer downstream automation
- use `include_polygons=false` when payload size matters more than geometry fidelity

## 10. Versioning / Stability Notes

- The OCR API uses path-based versioning: `/v1/...`
- Current documented OCR contract:
  - `GET /v1/ocr/backend`
  - `POST /v1/ocr/page`
- No authentication layer is documented because none exists today.
- The current goal is contract stability for existing OCR endpoints; future OCR capabilities should be added as new endpoints rather than changing the current payload shapes.
