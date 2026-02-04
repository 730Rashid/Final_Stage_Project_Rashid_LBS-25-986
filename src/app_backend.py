"""
App Backend for Semantic Search and Classification.

This module provides the AI logic for the CrisisMMD visualisation app:
- Data loading (metadata, embeddings)
- CLIP model management
- Semantic search (text-to-image)
- Visual search (image-to-image)
- Zero-shot classification

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import numpy as np
import torch
import json
import threading
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from utils.event_utils import EVENT_MAPPINGS, parse_event


# Path Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "visualisation" / "umap_data.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy"
IMAGE_FOLDER = PROJECT_ROOT / "data" / "processed" / "clean_data"


# Classification Labels
CLASSIFICATION_LABELS = [
    "fire or flames",
    "flood or water damage",
    "damaged building or infrastructure",
    "rescue operation",
    "debris or rubble",
    "vehicle",
    "people or crowd",
    "smoke",
    "fallen trees",
    "emergency services"
]

LABEL_DISPLAY_NAMES = {
    "fire or flames": "Fire",
    "flood or water damage": "Flood",
    "damaged building or infrastructure": "Damage",
    "rescue operation": "Rescue",
    "debris or rubble": "Debris",
    "vehicle": "Vehicle",
    "people or crowd": "People",
    "smoke": "Smoke",
    "fallen trees": "Trees",
    "emergency services": "Emergency"
}


class CrisisDataManager:
    """
    Manages crisis image data and AI models.
    
    This class handles:
    - Loading metadata and embeddings
    - CLIP model for text/image encoding
    - Semantic and visual search
    - Zero-shot classification
    """
    
    def __init__(self):
        """Initialise the data manager."""
        self.df = None
        self.embeddings = None
        self.unique_events = None
        self.clip_model = None
        self.clip_processor = None
        self.label_embeddings = None
        self.device = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load all data and models."""
        if self._loaded:
            return True
        
        print("Starting Application...")
        
        # Load metadata
        if not self._load_metadata():
            return False
        
        # Load CLIP model
        if not self._load_clip_model():
            return False
        
        # Load embeddings
        if not self._load_embeddings():
            return False
        
        # Precompute label embeddings
        self._precompute_label_embeddings()
        
        self._loaded = True
        return True
    
    def _load_metadata(self) -> bool:
        """Load UMAP metadata."""
        try:
            with open(DATA_PATH, "r") as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
            
            # Add derived columns
            self.df["event"] = self.df["path"].apply(parse_event)
            self.df["filename"] = self.df["path"].apply(lambda p: Path(p).name)
            self.df["hover"] = self.df.apply(
                lambda r: "<b>{}</b><br>{}".format(r["event"], r["filename"]),
                axis=1
            )
            self.df["original_idx"] = self.df.index
            
            self.unique_events = sorted(self.df["event"].unique())
            
            print("Metadata loaded: {} images across {} events".format(
                len(self.df), len(self.unique_events)
            ))
            return True
            
        except FileNotFoundError:
            print("Could not find {}. Run umap_reduction.py first.".format(DATA_PATH))
            return False
    
    def _load_clip_model(self) -> bool:
        """Load CLIP model for encoding."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print("Loading CLIP Model...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print("Inference Device: {}".format(self.device))
            
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            return True
            
        except Exception as e:
            print("Failed to load CLIP model: {}".format(e))
            return False
    
    def _load_embeddings(self) -> bool:
        """Load image embeddings."""
        try:
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 512:
                print("Error: Expected embeddings shape (N, 512), got {}".format(
                    self.embeddings.shape
                ))
                return False
            print("Embeddings loaded: {}".format(self.embeddings.shape))
            return True
        except Exception as e:
            print("Failed to load embeddings: {}".format(e))
            return False
    
    def _precompute_label_embeddings(self):
        """Precompute embeddings for classification labels."""
        try:
            print("Precomputing label embeddings...")
            
            label_inputs = self.clip_processor(
                text=CLASSIFICATION_LABELS,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                label_features = self.clip_model.get_text_features(**label_inputs)
            
            if hasattr(label_features, "pooler_output"):
                label_features = label_features.pooler_output
            elif hasattr(label_features, "last_hidden_state"):
                label_features = label_features.last_hidden_state[:, 0, :]
            
            label_features = label_features / label_features.norm(p=2, dim=-1, keepdim=True)
            self.label_embeddings = label_features.cpu().numpy()
            
            print("Label embeddings ready")
            
        except Exception as e:
            print("Could not precompute label embeddings: {}".format(e))
            self.label_embeddings = None
    
    def semantic_search(
        self,
        query: str,
        subset_indices: Optional[List[int]] = None,
        top_k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find images matching the text query.
        
        Args:
            query: Natural language query.
            subset_indices: Optional list of indices to search within.
            top_k: Number of results to return.
            
        Returns:
            Tuple of (indices, similarity_scores).
        """
        # Encode query
        inputs = self.clip_processor(
            text=[query],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        elif hasattr(text_features, "last_hidden_state"):
            text_features = text_features.last_hidden_state[:, 0, :]
        
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_vector = text_features.cpu().numpy()
        
        # Search
        if subset_indices is not None and len(subset_indices) > 0:
            subset_embeddings = self.embeddings[subset_indices]
            similarities = cosine_similarity(text_vector, subset_embeddings)[0]
            local_top_k = min(top_k, len(subset_indices))
            local_top_indices = np.argsort(similarities)[::-1][:local_top_k]
            global_indices = np.array(subset_indices)[local_top_indices]
            return global_indices, similarities[local_top_indices]
        else:
            similarities = cosine_similarity(text_vector, self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return top_indices, similarities[top_indices]
    
    def visual_search(
        self,
        image_index: int,
        subset_indices: Optional[List[int]] = None,
        top_k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find visually similar images.
        
        Args:
            image_index: Index of the query image.
            subset_indices: Optional list of indices to search within.
            top_k: Number of results to return.
            
        Returns:
            Tuple of (indices, similarity_scores).
        """
        query_vector = self.embeddings[image_index].reshape(1, -1)
        
        if subset_indices is not None and len(subset_indices) > 0:
            subset_embeddings = self.embeddings[subset_indices]
            similarities = cosine_similarity(query_vector, subset_embeddings)[0]
            local_top_k = min(top_k, len(subset_indices))
            local_top_indices = np.argsort(similarities)[::-1][:local_top_k]
            global_indices = np.array(subset_indices)[local_top_indices]
            return global_indices, similarities[local_top_indices]
        else:
            similarities = cosine_similarity(query_vector, self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return top_indices, similarities[top_indices]
    
    def classify_image(
        self,
        image_index: int,
        threshold: float = 0.20
    ) -> List[Tuple[str, float]]:
        """
        Classify image content using zero-shot classification.
        
        Args:
            image_index: Index of the image to classify.
            threshold: Minimum confidence threshold.
            
        Returns:
            List of (label, confidence) tuples.
        """
        if self.label_embeddings is None:
            return []
        
        image_vector = self.embeddings[image_index].reshape(1, -1)
        similarities = cosine_similarity(image_vector, self.label_embeddings)[0]
        
        results = []
        for i, score in enumerate(similarities):
            if score >= threshold:
                display_name = LABEL_DISPLAY_NAMES[CLASSIFICATION_LABELS[i]]
                results.append((display_name, float(score)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# Global instance
_manager = None


def get_manager() -> CrisisDataManager:
    """Get or create the global data manager."""
    global _manager
    if _manager is None:
        _manager = CrisisDataManager()
        _manager.load()
    return _manager


# Convenience exports
def get_dataframe():
    """Get the metadata DataFrame."""
    return get_manager().df


def get_embeddings():
    """Get the image embeddings."""
    return get_manager().embeddings


def get_unique_events():
    """Get list of unique events."""
    return get_manager().unique_events


def semantic_search(query, subset_indices=None, top_k=50):
    """Convenience wrapper for semantic search."""
    return get_manager().semantic_search(query, subset_indices, top_k)


def visual_search(image_index, subset_indices=None, top_k=50):
    """Convenience wrapper for visual search."""
    return get_manager().visual_search(image_index, subset_indices, top_k)


def classify_image(image_index, threshold=0.20):
    """Convenience wrapper for image classification."""
    return get_manager().classify_image(image_index, threshold)


if __name__ == "__main__":
    # Demo usage
    print("\nCrisisMMD Backend Demo\n")
    
    manager = get_manager()
    
    print("Dataset: {} images".format(len(manager.df)))
    print("Events: {}".format(manager.unique_events))
    
    # Test search
    query = "flooded street"
    print("\nSearch: '{}'".format(query))
    indices, scores = manager.semantic_search(query, top_k=5)
    
    for idx, score in zip(indices, scores):
        print("  [{:.1f}%] {}".format(score * 100, manager.df.iloc[idx]["filename"]))
