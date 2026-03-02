---
name: sam-segment-llm-caller
description: Generate, validate, and optimize calls to the SAM Finds segmentation API (`POST /v1/sam/segment`). Use when Codex must turn natural-language intent into executable API calls (curl, JavaScript, Python, or backend code), choose the right prompt mode (`text`, `box`, or `points`), and tune `output`, `multimask_output`, and `max_masks` for best latency/throughput vs quality.
---

# SAM Segment LLM Caller

## Objective
Turn user intent into a correct, high-performance request to `/v1/sam/segment`, then parse and return the requested result format.

## Workflow
1. Gather required inputs: image source, object/task intent, and whether caller needs full masks or just locations.
2. Convert image bytes to base64 PNG/JPEG and set `image`.
3. Set exactly one prompt type in `prompt`.
4. Choose a performance profile and map it to `output`, `multimask_output`, and `max_masks`.
5. Generate executable request code (not pseudocode).
6. Validate response shape by `output` mode and handle known API errors.

## Request Construction Rules
- Send `POST /v1/sam/segment` with `Content-Type: application/json`.
- Set `image` to base64-encoded PNG or JPEG bytes.
- Set exactly one of:
  - `prompt.text`: default for most use cases.
  - `prompt.box`: use when caller already has a tight object box.
  - `prompt.points`: use only when foreground/background clicks are the only input.
- Never send multiple prompt types in one request.
- Choose `output` intentionally:
  - `masks`: full COCO RLE masks.
  - `points`: centroid points only (smaller response, faster downstream handling).

## Performance Tuning
Choose one profile unless user explicitly requests different tradeoffs.

| Profile | Use when | output | multimask_output | max_masks |
|---|---|---|---|---|
| Fast | Real-time UX, previews, high QPS | `points` | `false` | `1` |
| Balanced | Typical app behavior | `masks` | `false` | `1` or `2` |
| Quality | Need alternate candidates / highest recall | `masks` | `true` | `3` |

Additional performance rules:
- Prefer `points` output when masks are not strictly required.
- Set `multimask_output=false` unless multiple hypotheses are needed.
- Keep `max_masks` as low as possible.
- Prefer concise, specific `text` prompts.
- Use `box` prompts when high-quality geometry is already available.

## LLM Output Pattern
When user asks to "call the API", produce:
1. One runnable snippet in the requested language.
2. The exact JSON payload used.
3. A one-line rationale for chosen performance profile.

If the user does not specify a profile, default to `Balanced`.

## Minimal Payload Templates

### Balanced (default)
```json
{
  "image": "<base64>",
  "prompt": { "text": "dog" },
  "multimask_output": false,
  "max_masks": 2,
  "output": "masks"
}
```

### Fast
```json
{
  "image": "<base64>",
  "prompt": { "text": "dog" },
  "multimask_output": false,
  "max_masks": 1,
  "output": "points"
}
```

## Error Handling Contract
Handle these API errors explicitly:
- `INVALID_IMAGE` (400): base64/image decode issue.
- `INVALID_PROMPT` (400): malformed prompt or multiple prompt types.
- `EMPTY_RESULT` (400): no masks returned.
- `MODEL_ERROR` (500): inference failure.

Retry only for transient server failures (`MODEL_ERROR`), not for invalid input errors.

## Response Parsing Rules
- If `output="masks"`, read `response.masks[].mask_rle` and `confidence`.
- If `output="points"`, read `response.points[].point` and `confidence`.
- Use `response.meta` for telemetry (`prompt_type`, dimensions, and multimask mode).

## References
- Read [references/segment-api.md](references/segment-api.md) for endpoint field details and language-specific request snippets.
