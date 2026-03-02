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

Request body:
```json
{
  "image": "<base64 PNG/JPEG>",
  "text": "<object description>"
}
```

Do not use:
- `prompt.box`
- `prompt.points`
- output mode selection

The backend already enforces balanced behavior and returns points for click workflows.

## Workflow
1. Determine the exact target object from user intent.
2. Write a precise `text` description.
3. Generate runnable request code (curl, JavaScript, or Python).
4. Parse `points[0].point` as the click target.

## Prompt Wording Rules
Use these rules to maximize hit rate:
- Start with the object noun: `backpack`, `dog`, `blue button`.
- Add one or two discriminators: color, size, side, relation.
- Keep phrases short and concrete.
- Avoid coordinates, pixel values, and geometry instructions.
- Avoid multi-object requests in one prompt.

Good examples:
- `red coffee mug near the keyboard`
- `leftmost person wearing a white shirt`
- `blue submit button at the bottom`

Weak examples:
- `object at x=420 y=300`
- `find everything important`
- `person and bike and car`

## LLM Output Pattern
When user asks to call the API, return:
1. One executable code snippet in requested language.
2. The exact JSON payload.
3. One-line explanation of why the `text` phrasing should disambiguate the target.

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
