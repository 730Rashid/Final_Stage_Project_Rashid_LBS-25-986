"""
CLIP Patch Relevance Heatmaps for Explainable AI.

Computes a spatial relevance map by measuring the cosine similarity between
each image patch's hidden state and the CLS token's hidden state in CLIP's
ViT-B/32 vision encoder. Patches that are highly similar to the global image
representation (CLS) are considered the most "attended to" by the model.

This approach is robust to all attention implementations (including SDPA in
newer transformers versions) since it only requires the final hidden states,
not raw attention weight tensors.

ViT-B/32 geometry:
  - Input: 224×224
  - Patch size: 32×32
  - Patches: 7×7 = 49 spatial tokens + 1 CLS token = 50 total tokens
  - Hidden dim: 768

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import numpy as np
import torch
import cv2
from PIL import Image


class CLIPAttentionRollout:
    """
    Computes patch-CLS relevance heatmaps for CLIP ViT-B/32 images.

    For each image patch in the final hidden layer, the cosine similarity to
    the CLS token is computed. High similarity means that patch contributed
    strongly to the global image representation — i.e. where CLIP was focusing.

    Works with all HuggingFace transformers versions regardless of attention
    implementation (eager, SDPA, Flash Attention).
    """

    PATCH_GRID = 7      # sqrt(49 patches) for ViT-B/32
    BLEND_ALPHA = 0.55  # heatmap weight in the alpha blend

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self._cache: dict = {}  # path -> JPEG bytes

    def compute(self, image_path: str) -> bytes:
        """Compute and return JPEG bytes of the heatmap blended onto the image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            JPEG bytes of the relevance heatmap overlaid on the original image.
        """
        if image_path in self._cache:
            return self._cache[image_path]

        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)

        inputs = self.processor(
            images=img_pil,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )

        # last_hidden_state: (1, 50, 768) — token 0 is CLS, tokens 1–49 are patches
        hidden = vision_outputs.last_hidden_state[0]  # (50, 768)

        cls_vec   = hidden[0:1]     # (1, 768)
        patch_vec = hidden[1:]      # (49, 768)

        # Cosine similarity: each patch vs the CLS token
        cls_norm   = cls_vec   / (cls_vec.norm(dim=-1, keepdim=True)   + 1e-8)
        patch_norm = patch_vec / (patch_vec.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (patch_norm @ cls_norm.T).squeeze().cpu().numpy()  # (49,)

        # Normalize to [0, 1]
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

        # Reshape to 7×7 spatial grid and resize to original image dimensions
        grid = sim.reshape(self.PATCH_GRID, self.PATCH_GRID)
        h, w = img_np.shape[:2]
        mask = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        mask = np.clip(mask, 0, 1)

        # Apply JET colormap and blend with original
        heatmap_bgr = cv2.applyColorMap(
            (mask * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        blended = (
            self.BLEND_ALPHA * heatmap_rgb.astype(np.float32) +
            (1.0 - self.BLEND_ALPHA) * img_np.astype(np.float32)
        ).astype(np.uint8)

        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(
            ".jpg", blended_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not success:
            raise RuntimeError("Failed to encode heatmap as JPEG")

        result_bytes = buffer.tobytes()
        self._cache[image_path] = result_bytes
        return result_bytes
