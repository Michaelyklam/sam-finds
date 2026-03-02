---
name: sam-segment-llm-caller
description: Generate robust calls to the LLM-safe SAM Finds endpoint (`POST /v1/sam/segment/text-points`) and optimize text prompt wording for accurate clickable results. Use when Codex needs to turn user intent into segmentation request code and improve object descriptions so the returned point is reliably usable for click automation.
---

# SAM Segment LLM Caller

## Objective
Turn natural-language intent into a high-quality text prompt and call `/v1/sam/segment/text-points`.

## Fixed API Contract
Use only this endpoint for LLM-generated calls:
- `POST /v1/sam/segment/text-points`
- URL for devices on the same local network: `http://192.168.0.10:8000/v1/sam/segment/text-points`

Request body:
```json
{
  "image": "<base64 PNG/JPEG>",
  "text": "<object description>"
}
```

This endpoint only accepts `image` and `text`, and always returns point output with server-enforced balanced settings for click workflows.
Use one shared base URL for all clients: `http://192.168.0.10:8000`.

Example with a concrete host IP:
- `http://192.168.0.10:8000/v1/sam/segment/text-points`

## Workflow
1. Determine the exact target object from user intent.
2. Write a precise `text` description.
3. Generate runnable request code (curl, JavaScript, or Python).
4. Parse `points[0].point` as the click target.

## Prompt Wording Rules
Use these rules to maximize hit rate:
- Start with the object noun: `mug`, `camera`, `mouse`, `duct tape`.
- Add one or two intrinsic discriminators: color, material, shape, brand-like appearance.
- Keep phrases short and concrete.
- Avoid relational or scene-reasoning language (`near`, `next to`, `on top of`, `left of`, `right of`).
- Avoid coordinates, pixel values, and geometry instructions.
- Avoid multi-object requests in one prompt.

Good examples:
- `gray roll of duct tape`
- `black mug`
- `white computer mouse`
- `black camera`
- `red X button`

Weak examples:
- `red mug near the keyboard`
- `object on the left side`
- `person next to the bike`
- `object at x=420 y=300`
- `find everything important`
- `person and bike and car`

## LLM Output Pattern
When user asks to call the API, return:
1. One executable code snippet in requested language.
2. The exact JSON payload.
3. One-line explanation of why the attribute-based `text` phrasing should disambiguate the target.

## Response Handling
Read the first returned point for click automation:
- `response.points[0].point.x`
- `response.points[0].point.y`

Handle errors explicitly:
- `INVALID_IMAGE` (400)
- `EMPTY_RESULT` (400)
- `MODEL_ERROR` (500)

## References
- Read [references/segment-api.md](references/segment-api.md) for request/response examples.
