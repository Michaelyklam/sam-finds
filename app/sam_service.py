from __future__ import annotations

import logging
import os

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
        from sam3.model_builder import build_sam3_image_model  # type: ignore[import-untyped]
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-untyped]

        ckpt = os.environ.get("SAM3_CHECKPOINT")
        logger.info("Loading SAM3 model (checkpoint=%s)…", ckpt)
        self.model = build_sam3_image_model(
            checkpoint_path=ckpt,
            load_from_HF=ckpt is None,
        )
        self.processor = Sam3Processor(self.model, confidence_threshold=0.3)
        logger.info("SAM3 model loaded")

    def predict(
        self,
        image: Image.Image,
        prompt: Prompt,
        *,
        multimask_output: bool = True,
        max_masks: int = 3,
    ) -> list[MaskResult]:
        assert self.processor is not None

        width, height = image.size
        state = self.processor.set_image(image)

        try:
            if prompt.text is not None:
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(
                    state=state, prompt=prompt.text
                )
            elif prompt.box is not None:
                b = prompt.box
                # Convert x1,y1,x2,y2 to normalized cxcywh
                cx = (b.x1 + b.x2) / 2.0 / width
                cy = (b.y1 + b.y2) / 2.0 / height
                bw = (b.x2 - b.x1) / width
                bh = (b.y2 - b.y1) / height
                norm_box = [cx, cy, bw, bh]

                self.processor.reset_all_prompts(state)
                state = self.processor.add_geometric_prompt(
                    state=state, box=norm_box, label=True
                )
            elif prompt.points is not None:
                # Convert point prompts to small box prompts
                # (Sam3Processor doesn't have native point support)
                self.processor.reset_all_prompts(state)
                for pt in prompt.points:
                    sz = 10
                    cx = pt.x / width
                    cy = pt.y / height
                    bw = (sz * 2) / width
                    bh = (sz * 2) / height
                    norm_box = [cx, cy, bw, bh]
                    state = self.processor.add_geometric_prompt(
                        state=state, box=norm_box, label=bool(pt.label)
                    )
            else:
                raise SAMError(INVALID_PROMPT, "No prompt provided")
        except SAMError:
            raise
        except Exception as exc:
            raise SAMError(MODEL_ERROR, f"Model inference failed: {exc}", 500)

        # Extract results from inference state
        masks_tensor = state.get("masks")
        scores_tensor = state.get("scores")

        if masks_tensor is None or len(masks_tensor) == 0:
            raise SAMError(EMPTY_RESULT, "Model returned no masks")

        # Sort by confidence descending and truncate
        scores_np = scores_tensor.cpu().numpy() if hasattr(scores_tensor, 'cpu') else np.array(scores_tensor)
        masks_np = masks_tensor.cpu().numpy() if hasattr(masks_tensor, 'cpu') else np.array(masks_tensor)

        order = np.argsort(scores_np)[::-1][:max_masks]

        results: list[MaskResult] = []
        for rank, idx in enumerate(order):
            # masks shape: [N, 1, H, W] — squeeze the channel dim
            binary = masks_np[idx].squeeze(0).astype(np.uint8)
            binary = np.asfortranarray(binary)
            rle = mask_utils.encode(binary)
            results.append(
                MaskResult(
                    id=str(rank),
                    confidence=round(float(scores_np[idx]), 4),
                    mask_rle=MaskRLE(
                        counts=rle["counts"].decode("utf-8"),
                        size=list(rle["size"]),
                    ),
                )
            )

        return results
