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
    Input: 224x224
    Patch size: 32x32
    Patches: 7x7 = 49 spatial tokens + 1 CLS token = 50 total tokens
    Hidden dim: 768

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
    strongly to the global image representation, i.e. where CLIP was focusing.

    Works with all HuggingFace transformers versions regardless of attention
    implementation (eager, SDPA, Flash Attention).
    """

    PATCH_GRID = 7      # sqrt(49 patches) for ViT-B/32
    BLEND_ALPHA = 0.55  # heatmap weight in the alpha blend
    BAR_HEIGHT = 30     # colour bar height in pixels
    LABEL_MARGIN = 30   # space for text labels below the bar

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self._cache: dict = {}  # path to JPEG bytes, stats dict

    def _compute_patch_relevance(self, image_path: str):
        """Run CLIP forward pass and return raw similarity grid + image array.

        Returns:
            (img_np, sim, grid) where sim is the flat (49,) normalised similarity
            and grid is the (7,7) reshaped version.
        """
        
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)

        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )

        hidden = vision_outputs.last_hidden_state[0]

        cls_vec = hidden[0:1]     
        patch_vec = hidden[1:]      

        # Cosine similarity each patch vs the CLS token
        
        cls_norm = cls_vec / (cls_vec.norm(dim=-1, keepdim=True) + 1e-8)
        
        patch_norm = patch_vec / (patch_vec.norm(dim=-1, keepdim=True) + 1e-8)
        
        sim = (patch_norm @ cls_norm.T).squeeze().cpu().numpy()  

        # Normalise it to [0, 1]
        
        sim_min, sim_max = float(sim.min()), float(sim.max())
        sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-8)

        grid = sim_norm.reshape(self.PATCH_GRID, self.PATCH_GRID)
        
        

        return img_np, sim_norm, grid, sim_min, sim_max



    def _build_stats(self, sim, grid, sim_min, sim_max):
        """Compute human-readable attention statistics from the similarity values."""
        # Identify which region of the 7x7 grid has the highest attention
        
        peak_row, peak_col = np.unravel_index(np.argmax(grid), grid.shape)
        
        row_label = ["top", "centre", "bottom"][min(peak_row * 3 // self.PATCH_GRID, 2)]
        col_label = ["left", "centre", "right"][min(peak_col * 3 // self.PATCH_GRID, 2)]
        
        if row_label == "centre" and col_label == "centre":
            focus_region = "centre"
            
        else:
            focus_region = "{}-{}".format(row_label, col_label)

        # Basically how concentrated is the attention? entropy based
        # Low entropy = concentrated on few patches, high = spread evenly
        
        p = sim / (sim.sum() + 1e-8)
        entropy = -np.sum(p * np.log(p + 1e-10))
        max_entropy = np.log(len(sim))
        concentration = 1.0 - (entropy / max_entropy)  # 0 = uniform and 1 = single patch

        # What percentage of total attention is in the top 25% of patches
        
        threshold = np.percentile(sim, 75)
        top_quarter_share = float(sim[sim >= threshold].sum() / (sim.sum() + 1e-8))
        

        return {
            "focus_region": focus_region,
            "peak_value": float(grid.max()),
            "mean_attention": float(sim.mean()),
            "concentration": float(concentration),
            "top_quarter_share": round(top_quarter_share * 100, 1),
            "raw_sim_range": [round(sim_min, 4), round(sim_max, 4)],
        }

    def _draw_colour_bar(self, canvas_bgr, img_width):
        """Draw a horizontal JET colour bar with Low/High labels at the bottom."""
        
        bar_top = canvas_bgr.shape[0] - self.BAR_HEIGHT - self.LABEL_MARGIN
        bar_bottom = bar_top + self.BAR_HEIGHT

        margin_x = max(30, img_width // 10)
        
        bar_left = margin_x
        bar_right = img_width - margin_x
        bar_width = bar_right - bar_left

        # Build a 1-pixel-tall gradient and resize to the bar dimensions
        
        gradient = np.linspace(0, 255, bar_width).astype(np.uint8).reshape(1, -1)
        gradient_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        gradient_bar = cv2.resize(gradient_bar, (bar_width, self.BAR_HEIGHT))

        # Paste gradient bar onto the canvas
        canvas_bgr[bar_top:bar_bottom, bar_left:bar_right] = gradient_bar

        # Draw a thin border around the bar
        cv2.rectangle(canvas_bgr, (bar_left, bar_top), (bar_right - 1, bar_bottom - 1), (180, 180, 180), 1)

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        font_thick = 1
        label_y = bar_bottom + 20

        cv2.putText(canvas_bgr, "Low Attention", (bar_left, label_y), font, font_scale, (200, 200, 200), font_thick, cv2.LINE_AA)

        high_text = "High Attention"
        
        (tw, _), _ = cv2.getTextSize(high_text, font, font_scale, font_thick)
        cv2.putText(canvas_bgr, high_text, (bar_right - tw, label_y), font, font_scale, (200, 200, 200), font_thick, cv2.LINE_AA)

        return canvas_bgr
    
    

    def compute(self, image_path: str) -> bytes:
        """Compute and return JPEG bytes of the heatmap blended onto the image.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            JPEG bytes of the relevance heatmap overlaid on the original image,
            including a colour-bar legend.
        """
        
        if image_path in self._cache:
            return self._cache[image_path][0]

        img_np, sim, grid, sim_min, sim_max = self._compute_patch_relevance(image_path)
        stats = self._build_stats(sim, grid, sim_min, sim_max)

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
            self.BLEND_ALPHA * heatmap_rgb.astype(np.float32) + (1.0 - self.BLEND_ALPHA) * img_np.astype(np.float32)
        ).astype(np.uint8)

        # Create canvas with extra space for the colour bar legend
        
        extra_h = self.BAR_HEIGHT + self.LABEL_MARGIN + 15
        canvas = np.zeros((h + extra_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = blended

        # Fill legend area with dark background
        canvas[h:, :] = 30

        # Convert to BGR for OpenCV drawing then add colour bar
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        canvas_bgr = self._draw_colour_bar(canvas_bgr, w)

        success, buffer = cv2.imencode(".jpg", canvas_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        
        if not success:
            raise RuntimeError("Failed to encode heatmap as JPEG")

        result_bytes = buffer.tobytes()
        self._cache[image_path] = (result_bytes, stats)

        return result_bytes


    def compute_with_stats(self, image_path: str) -> tuple:
        """Compute heatmap and return (JPEG bytes, stats dict).

        Stats dict contains:
            focus_region: str - where the model focused most (e.g. "top-left")
            peak_value: float - max normalised attention value
            mean_attention: float - average patch attention
            concentration: float - 0 = uniform, 1 = single-patch focus
            top_quarter_share: float - % of attention in top 25% of patches
            raw_sim_range: [min, max] - raw cosine similarity range before normalisation
        """
        
        if image_path not in self._cache:
            self.compute(image_path)
            
        return self._cache[image_path]
