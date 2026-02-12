"""
CLIP Interrogation Captioning for Crisis Images.

This class uses CLIP's discriminative capabilities to generate natural
captions for crisis images. Instead of a separate captioning model, we probe
CLIP with structured vocabularies across multiple categories and compose
human-readable descriptions.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class CLIPInterrogator:
    """
    Generate natural captions for crisis images using CLIP interrogation.

    This class contains CLIP with structured vocabularies across 5 categories:
    - Scene type (what kind of disaster or location)
    - Damage level (severity of impact)
    - Objects present (what's visible in the image)
    - Weather conditions (environmental context)
    - Human activity (people and their actions)

    The interrogator composes natural captions that sound human and provide
    meaningful context for humanitarian response.
    """

    # Structured vocabulary across 5 categories
    # Each phrase is crafted to sound natural when combined into a sentence
    VOCABULARIES = {
        "scene_type": [
            "urban area",
            "residential neighbourhood",
            "commercial district",
            "rural area",
            "coastal region",
            "mountainous terrain",
            "forest area",
            "agricultural land",
            "industrial zone",
            "highway or road",
            "bridge or overpass",
            "river or waterway",
            "makeshift shelter area",
            "evacuation site",
            "temporary camp"
        ],
        "damage_level": [
            "severe structural damage",
            "moderate damage",
            "minor damage",
            "completely destroyed",
            "partially collapsed",
            "intact but affected",
            "flooded area",
            "burnt or charred",
            "debris scattered",
            "structural cracks visible"
        ],
        "objects_present": [
            "damaged buildings",
            "collapsed structures",
            "emergency vehicles",
            "rescue equipment",
            "debris and rubble",
            "fallen trees",
            "damaged vehicles",
            "temporary shelters",
            "aid supplies",
            "medical equipment",
            "power lines down",
            "water damage",
            "smoke or fire",
            "flood water",
            "rescue boats"
        ],
        "weather_conditions": [
            "clear skies",
            "overcast conditions",
            "heavy rain",
            "stormy weather",
            "dusty atmosphere",
            "smoky conditions",
            "foggy conditions",
            "bright daylight",
            "low light conditions",
            "nighttime scene"
        ],
        "human_activity": [
            "rescue operations underway",
            "people evacuating",
            "emergency responders present",
            "crowd gathered",
            "victims seeking help",
            "aid distribution",
            "medical assistance",
            "search and rescue",
            "people sheltering",
            "community response",
            "no visible people",
            "bystanders observing"
        ]
    }

    def __init__(self, clip_model, clip_processor, device):
        """
        We initialise the CLIP interrogator.

        Args:
            clip_model: Pretrained CLIP model instance
            clip_processor: CLIP processor for text encoding
            device: Device to run inference on (cuda/cpu)
        """
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

        # Will store precomputed embeddings for each category
        self.text_embeddings = {}
        self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        """
        Precompute text embeddings for all vocabulary phrases.

        This is done once at initialisation to make interrogation fast.
        We encode all ~80 phrases in about 1 second.
        """
        print("Precomputing text embeddings for CLIP captioning...")

        try:
            for category, phrases in self.VOCABULARIES.items():
                # Encode all phrases in this category
                inputs = self.clip_processor(
                    text=phrases,
                    return_tensors="pt",
                    padding=True
                ).to(self.device) # Set to GPU for faster compute

                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)

                # Handle different CLIP output formats
                if hasattr(text_features, "pooler_output"):
                    text_features = text_features.pooler_output
                elif hasattr(text_features, "last_hidden_state"):
                    text_features = text_features.last_hidden_state[:, 0, :]

                # Normalise embeddings for cosine similarity use
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                # Store as numpy array with the phrases for easy lookup
                self.text_embeddings[category] = {
                    "embeddings": text_features.cpu().numpy(),
                    "phrases": phrases
                }

            print("Text embeddings ready for {} categories".format(len(self.VOCABULARIES)))

        except Exception as e:
            print("Failed to precompute text embeddings: {}".format(e))
            # Clear embeddings if something went wrong to free up memory
            self.text_embeddings = {}

    def interrogate(
        self,
        image_embedding: np.ndarray,
        confidence_threshold: float = 0.22
    ) -> Dict[str, Tuple[str, float]]:
        """
        Interrogate an image embedding across all categories.

        Args:
            image_embedding: 512D CLIP embedding of the image
            confidence_threshold: Minimum confidence to include a match

        Returns:
            Dictionary mapping category names to (best_phrase, confidence) tuples

        Example:
            {
                "scene_type": ("urban area", 0.45),
                "damage_level": ("severe structural damage", 0.38),
                ...
            }
        """
        if not self.text_embeddings:
            return {}

        results = {}

        # Ensure embedding is the right shape
        if image_embedding.ndim == 1:
            image_embedding = image_embedding.reshape(1, -1)

        # Compare against each category's vocabulary
        for category, data in self.text_embeddings.items():
            embeddings = data["embeddings"]
            phrases = data["phrases"]

            # Compute cosine similarity with all phrases in this category
            similarities = cosine_similarity(image_embedding, embeddings)[0]

            # Find the best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            # Only include if confidence is above threshold
            if best_score >= confidence_threshold:
                results[category] = (phrases[best_idx], float(best_score))

        return results

    def compose_caption(
        self,
        interrogation_result: Dict[str, Tuple[str, float]],
        style: str = "natural"
    ) -> str:
        """
        Compose a natural-sounding caption from interrogation results.

        Args:
            interrogation_result: Output from interrogate()
            style: Caption style - "natural" (full sentence) or "brief" (short)

        Returns:
            Human-readable caption string
        """
        if not interrogation_result:
            return "Image shows a disaster scene."

        # Extract the matched phrases (ignore confidence scores for composition)
        scene = interrogation_result.get("scene_type", (None, 0))[0]
        damage = interrogation_result.get("damage_level", (None, 0))[0]
        objects = interrogation_result.get("objects_present", (None, 0))[0]
        weather = interrogation_result.get("weather_conditions", (None, 0))[0]
        activity = interrogation_result.get("human_activity", (None, 0))[0]

        if style == "brief":
            # Short, headline-style caption
            parts = []
            if scene:
                parts.append(scene)
            if damage:
                parts.append("with " + damage)

            if parts:
                return "; ".join(parts).capitalize() + "."
            return "Disaster scene."

        # Natural style sentence
        caption_parts = []

        # Start with the scene
        if scene:
            caption_parts.append("This appears to be a disaster in {}".format(
                "an " + scene if scene[0] in "aeiou" else "a " + scene
            ))
        else:
            caption_parts.append("This appears to be a disaster scene")

        # Add damage assessment
        if damage:
            caption_parts.append("showing {}".format(damage))

        # Add visible objects/elements
        if objects:
            caption_parts.append("with {} visible".format(objects))

        # Add weather context if notable
        if weather and weather not in ["clear skies", "bright daylight"]:
            caption_parts.append("under {}".format(weather))

        # Add human activity
        if activity and activity != "no visible people":
            # Make it flow naturally
            if "underway" in activity or "present" in activity:
                caption_parts.append("and {}".format(activity))
            else:
                caption_parts.append("with {}".format(activity))

        # Join parts with commas and periods
        caption = caption_parts[0]
        if len(caption_parts) > 1:
            caption += ", " + ", ".join(caption_parts[1:-1])
            if len(caption_parts) > 2:
                caption += ","
            caption += " " + caption_parts[-1]

        caption += "."

        return caption

    def batch_caption(
        self,
        image_embeddings: np.ndarray,
        confidence_threshold: float = 0.22,
        style: str = "natural"
    ) -> List[str]:
        """
        Generate captions for multiple images efficiently.

        Args:
            image_embeddings: (N, 512) array of CLIP embeddings
            confidence_threshold: Minimum confidence for phrase matching
            style: Caption style ("natural" or "brief")

        Returns:
            List of caption strings, one per image
        """
        captions = []

        for i in range(len(image_embeddings)):
            
            embedding = image_embeddings[i]
            interrogation = self.interrogate(embedding, confidence_threshold)
            caption = self.compose_caption(interrogation, style)
            captions.append(caption)

        return captions

    def get_detailed_breakdown(
        self,
        interrogation_result: Dict[str, Tuple[str, float]]
    ) -> Dict[str, Dict[str, any]]:
        """
        Get a detailed breakdown of interrogation results for UI display.

        Args:
            interrogation_result: Output from interrogate()

        Returns:
            Dictionary with category names, phrases, and confidence scores
            formatted for display
        """
        breakdown = {}

        # Friendly category names for display
        category_names = {
            "scene_type": "Scene Type",
            "damage_level": "Damage Assessment",
            "objects_present": "Visible Elements",
            "weather_conditions": "Weather Context",
            "human_activity": "Human Activity"
        }

        for category, (phrase, confidence) in interrogation_result.items():
            
            display_name = category_names.get(category, category)
            
            breakdown[display_name] = {
                "phrase": phrase.capitalize(),
                "confidence": confidence,
                "confidence_pct": int(confidence * 100)
            }

        return breakdown


# Convenience function for quick usage
def caption_image(
    image_embedding: np.ndarray,
    interrogator: CLIPInterrogator,
    style: str = "natural"
) -> Tuple[str, Dict]:
    """
    Quick function to caption a single image and get details.

    Args:
        image_embedding: 512D CLIP embedding
        interrogator: CLIPInterrogator instance
        style: Caption style ("natural" or "brief")

    Returns:
        Tuple of (caption_string, detailed_breakdown_dict)
    """
    interrogation = interrogator.interrogate(image_embedding)
    caption = interrogator.compose_caption(interrogation, style)
    breakdown = interrogator.get_detailed_breakdown(interrogation)

    return caption, breakdown


if __name__ == "__main__":
    print("\n Captioning using CLIP \n")
