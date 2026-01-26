"""
App Backend for Semantic Search.

This module provides the AI logic for text-to-image semantic search.
It loads pre-computed CLIP embeddings and allows searching for images
using natural language queries.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import sys
import numpy as np
import torch
import json
from pathlib import Path
from typing import List, Tuple, Optional


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Paths to pre-computed data
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy"
FILENAMES_PATH = PROJECT_ROOT / "data" / "embeddings" / "filenames.json"
UMAP_COORDS_PATH = PROJECT_ROOT / "data" / "visualisation" / "umap_coords.npy"


class SemanticSearchEngine:
    """
    Semantic search engine using CLIP embeddings.
    
    This class loads pre-computed image embeddings and provides
    text-to-image search using cosine similarity.
    """
    
    def __init__(self):
        """Initialise the search engine."""
        self.embeddings = None
        self.filenames = None
        self.umap_coords = None
        self.model = None
        self.processor = None
        self.device = None
        self._loaded = False
    
    def load(self):
        """Load embeddings and CLIP model."""
        if self._loaded:
            return True
        
        # Load embeddings
        if not EMBEDDINGS_PATH.exists():
            print("ERROR: Embeddings not found at {}".format(EMBEDDINGS_PATH))
            print("Run vectorise.py first")
            return False
        
        self.embeddings = np.load(EMBEDDINGS_PATH)
        print("Loaded {} embeddings".format(len(self.embeddings)))
        
        # Load filenames
        if FILENAMES_PATH.exists():
            with open(FILENAMES_PATH, "r") as f:
                self.filenames = json.load(f)
        
        # Load UMAP coordinates if available
        if UMAP_COORDS_PATH.exists():
            self.umap_coords = np.load(UMAP_COORDS_PATH)
            print("Loaded UMAP coordinates")
        
        # Load CLIP model for text encoding
        self._load_clip_model()
        
        self._loaded = True
        return True
    
    def _load_clip_model(self):
        """Load CLIP model for text encoding."""
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading CLIP model on {}...".format(self.device))
        
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model.eval()
        print("CLIP model loaded")
    
    def encode_text(self, query: str) -> np.ndarray:
        """
        Encode a text query into a CLIP vector.
        
        Args:
            query: Natural language query.
            
        Returns:
            512-dimensional normalised vector.
        """
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # Normalise for cosine similarity
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().flatten()
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float, str]]:
        """
        Search for images matching a text query.
        
        Args:
            query: Natural language query (e.g. "people in flood water").
            top_k: Number of results to return.
            
        Returns:
            List of (index, similarity_score, filepath) tuples.
        """
        if not self._loaded:
            self.load()
        
        # Encode the query
        query_vec = self.encode_text(query)
        
        # Compute cosine similarity (embeddings are already normalised)
        similarities = np.dot(self.embeddings, query_vec)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            filepath = self.filenames[idx] if self.filenames else str(idx)
            results.append((int(idx), score, filepath))
        
        return results
    
    def get_image_path(self, index: int) -> Optional[str]:
        """Get image path by index."""
        if self.filenames and 0 <= index < len(self.filenames):
            return self.filenames[index]
        return None
    
    def get_umap_coords(self, index: int) -> Optional[Tuple[float, float]]:
        """Get UMAP coordinates by index."""
        if self.umap_coords is not None and 0 <= index < len(self.umap_coords):
            return tuple(self.umap_coords[index])
        return None


# Global instance for easy access
_engine = None


def get_engine() -> SemanticSearchEngine:
    """Get or create the global search engine instance."""
    global _engine
    if _engine is None:
        _engine = SemanticSearchEngine()
    return _engine


def search(query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
    """
    Convenience function for semantic search.
    
    Args:
        query: Natural language query.
        top_k: Number of results.
        
    Returns:
        List of (index, score, filepath) tuples.
    """
    engine = get_engine()
    return engine.search(query, top_k)


if __name__ == "__main__":
    # Demo usage
    print("Semantic Search Demo")
    
    engine = get_engine()
    if engine.load():
        query = "people helping after flood"
        print("Query: '{}'".format(query))
        
        results = engine.search(query, top_k=5)
        
        print("Top 5 results:")
        for idx, score, path in results:
            print("  [{:.3f}] {}".format(score, Path(path).name))
