---
name: sam-segment-llm-caller
description: Generate robust calls to SAM Finds UI targeting endpoints and choose the correct path for named-button clicks vs object-based clicks. Use when Codex needs to call OCR+SAM endpoints for text-labeled controls or SAM-only endpoints for general object targeting.
---

# SAM Segment LLM Caller

## Objective
Turn natural-language intent into the right endpoint call for reliable UI clicking.

## Endpoint Selection
Use LAN base URL:
- `http://192.168.0.10:8000`

Use endpoints by task:
- **Named button/control text is important**: `POST /v1/ui/click-targets`
- **Need mask/points format with OCR assistance**: `POST /v1/ui/segment`
- **General object click (no text disambiguation needed)**: `POST /v1/sam/segment/text-points`

Use `text_mode` for text prompts:
- `screen_text`: OCR-assisted matching for visible labels.
- `visual`: plain SAM text prompting for visual object descriptions.
- `text_mode` is required in requests.

`/v1/ui/click-targets` request:
```json
{
  "image": "<base64 PNG/JPEG>",
  "target_text": "View Leaderboard button",
  "max_candidates": 3,
  "use_sam_refine": true
}
```

`/v1/ui/segment` keeps the existing prompt/output interaction format and returns `masks` or `points`.

## Workflow
1. Determine whether text-labeled controls must be disambiguated.
2. For named controls, use OCR-aware endpoint (`/v1/ui/click-targets` or `/v1/ui/segment`).
3. Write a precise target phrase.
3. Generate runnable request code (curl, JavaScript, or Python).
4. Parse point candidates and click top-ranked result.

## Prompt Wording Rules
Use these rules to maximize hit rate:
- Start with the object noun: `mug`, `camera`, `mouse`, `duct tape`.
- Add one or two intrinsic discriminators: color, material, shape, brand-like appearance.
- Keep phrases short and concrete.
- For button text targeting, use exact visible label text where possible.
- Avoid relational or scene-reasoning language (`near`, `next to`, `on top of`, `left of`, `right of`) as primary disambiguation.
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
For `/v1/ui/click-targets`, use:
- `response.candidates[0].point.x`
- `response.candidates[0].point.y`

For `/v1/ui/segment` or `/v1/sam/segment/text-points`, use:
- `response.points[0].point.x`
- `response.points[0].point.y`

Handle errors explicitly:
- `INVALID_IMAGE` (400)
- `EMPTY_RESULT` (400)
- `MODEL_ERROR` (500)
- `TEXT_NOT_FOUND` (400)
- `OCR_ERROR` (500)

## References
- Read [references/segment-api.md](references/segment-api.md) for request/response examples.
