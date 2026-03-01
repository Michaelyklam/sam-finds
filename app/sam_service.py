from __future__ import annotations

import base64
import io
import logging

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from app.errors import EMPTY_RESULT, INVALID_PROMPT, MODEL_ERROR, SAMError
from app.schemas import MaskResult, MaskRLE, Prompt

logger = logging.getLogger(__name__)


class SAMService:
    def __init__(self) -> None:
        self.model = None
        self.processor = None

    def load(self) -> None:
        from sam3 import build_sam3_image_model, Sam3Processor  # type: ignore[import-untyped]

        logger.info("Loading SAM3 model…")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        logger.info("SAM3 model loaded")

    def predict(
        self,
        image: Image.Image,
        prompt: Prompt,
        *,
        multimask_output: bool = True,
        max_masks: int = 3,
    ) -> list[MaskResult]:
        assert self.model is not None and self.processor is not None

        state = self.processor.set_image(image)

        kwargs: dict = {"multimask_output": multimask_output}
        if prompt.points is not None:
            kwargs["point_coords"] = np.array(
                [[p.x, p.y] for p in prompt.points]
            )
            kwargs["point_labels"] = np.array(
                [p.label for p in prompt.points]
            )
        elif prompt.box is not None:
            b = prompt.box
            kwargs["box"] = np.array([b.x1, b.y1, b.x2, b.y2])
        elif prompt.mask is not None:
            try:
                mask_bytes = base64.b64decode(prompt.mask)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
                kwargs["mask_input"] = np.array(mask_img)
            except Exception as exc:
                raise SAMError(INVALID_PROMPT, f"Invalid mask input: {exc}")
        else:
            raise SAMError(INVALID_PROMPT, "No prompt provided")

        try:
            masks, scores, _ = self.model.predict_inst(state, **kwargs)
        except Exception as exc:
            raise SAMError(MODEL_ERROR, f"Model inference failed: {exc}", 500)

        if masks is None or len(masks) == 0:
            raise SAMError(EMPTY_RESULT, "Model returned no masks")

        # Sort by confidence descending and truncate
        order = np.argsort(scores)[::-1][:max_masks]

        results: list[MaskResult] = []
        for rank, idx in enumerate(order):
            binary = np.asfortranarray(masks[idx].astype(np.uint8))
            rle = mask_utils.encode(binary)
            results.append(
                MaskResult(
                    id=str(rank),
                    confidence=round(float(scores[idx]), 4),
                    mask_rle=MaskRLE(
                        counts=rle["counts"].decode("utf-8"),
                        size=list(rle["size"]),
                    ),
                )
            )

        return results
