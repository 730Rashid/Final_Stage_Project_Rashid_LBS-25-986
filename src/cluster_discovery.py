"""
Unsupervised Cluster Discovery using HDBSCAN.

This script discovers semantic clusters in the CLIP embedding space without
relying on ground-truth labels. Each discovered cluster is automatically
named using CLIP text similarity to find the best description.

"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import hdbscan
from sklearn.preprocessing import normalize

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config
from transformers import CLIPProcessor, CLIPModel


# Paths
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy"
FILENAMES_PATH = PROJECT_ROOT / "data" / "embeddings" / "filenames.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "visualisation"


# Candidate labels for cluster naming
# These are semantic concepts that CLIP understands well
CANDIDATE_LABELS = [
    # Damage types
    "collapsed building",
    "flooded street",
    "fire damage",
    "structural damage",
    "debris and rubble",
    "destroyed infrastructure",
    "damaged vehicles",
    "fallen trees",
    
    # Scene types
    "aerial view of disaster",
    "street level damage",
    "residential area",
    "urban destruction",
    "rural flooding",
    "coastline damage",
    
    # Human elements
    "rescue operation",
    "people evacuating",
    "emergency responders",
    "people in boats",
    "crowd gathering",
    "volunteers helping",
    
    # Nature
    "wildfire smoke",
    "storm clouds",
    "floodwater",
    "mudslide",
    "earthquake cracks",
    
    # Media/Documents
    "news screenshot",
    "infographic",
    "map or diagram",
    "text overlay",
    "social media post",
    
    # Other
    "before and after comparison",
    "night scene",
    "daytime scene",
    "close-up damage",
    "wide shot destruction",
]


def load_data() -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Load CLIP embeddings from disk.
    
    Returns:
        Tuple of (embeddings array, filenames list) or (None, None) if not found.
    """
    
    print("Loading the embeddings")
    
    if not EMBEDDINGS_PATH.exists():
        print("Embeddings not found at {}".format(EMBEDDINGS_PATH))
        
        return None, None
    
    
    embeddings = np.load(EMBEDDINGS_PATH)
    
    print("Loaded {} embeddings of dimension {}".format(embeddings.shape[0], embeddings.shape[1]))
    
    filenames = None
    
    if FILENAMES_PATH.exists():
        
        with open(FILENAMES_PATH, "r") as f:
            filenames = json.load(f)
    
    return embeddings, filenames


def run_hdbscan(embeddings: np.ndarray) -> np.ndarray:
    """Run HDBSCAN clustering on the embeddings.
    
    Args:
        embeddings: Array of shape (N, 512).
        
    Returns:
        Array of cluster labels (shape N). -1 indicates noise.
    """
    
    print("Running HDBSCAN clustering")
    
    print("min_cluster_size: {}".format(config.HDBSCAN_MIN_CLUSTER_SIZE))
    print("min_samples: {}".format(config.HDBSCAN_MIN_SAMPLES))
    print("metric: {}".format(config.HDBSCAN_METRIC))
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        metric=config.HDBSCAN_METRIC,
        cluster_selection_method=config.HDBSCAN_CLUSTER_SELECTION_METHOD,
        core_dist_n_jobs=-1  # We should use all CPU cores
    )
    
    labels = clusterer.fit_predict(embeddings)
    
    # Statistics
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    print("Found {} clusters".format(n_clusters))
    print("Noise points: {} ({:.1f}%)".format(n_noise, 100 * n_noise / len(labels)))
    
    return labels


def compute_cluster_centroids(
    embeddings: np.ndarray, 
    labels: np.ndarray
) -> Dict[int, np.ndarray]:
    """Compute the centroid (mean embedding) for each cluster.
    
    Args:
        embeddings: Array of shape (N, 512).
        labels: Array of cluster labels.
        
    Returns:
        Dict mapping cluster ID to centroid embedding.
    """
    
    centroids = {}
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:
            continue  # Skip the noise
        
        mask = labels == label
        cluster_embeddings = embeddings[mask]
        centroid = cluster_embeddings.mean(axis=0)
        
        # L2 normalise for cosine similarity
        
        centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid
    
    return centroids


def load_clip_model() -> Tuple[CLIPModel, CLIPProcessor, str]:
    """Load CLIP model for text encoding.

    Returns:
        Tuple of (model, processor, device).
    """
    print("Loading CLIP model for cluster naming")
    
    if torch.cuda.is_available():
        device = "cuda"
        
    else:
        device = "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Device: {}".format(device))
    
    
    return model, processor, device


def encode_text_labels(
    labels: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str
) -> np.ndarray:
    """Encode text labels using CLIP.

    Args:
        labels: List of text descriptions.
        model: CLIP model.
        processor: CLIP processor.
        device: Device string.

    Returns:
        Array of shape (len(labels), 512) with text embeddings.
    """
    all_features = []
    
    for label in labels:
        inputs = processor(text=[label], return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            text_outputs = model.text_model(**inputs)
            text_features = model.text_projection(text_outputs.pooler_output)
            
        all_features.append(text_features.cpu().numpy())

    text_np = np.vstack(all_features)
    text_np = normalize(text_np, norm='l2', axis=1)
    
    return text_np


def name_clusters(
    centroids: Dict[int, np.ndarray],
    candidate_labels: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str
    
) -> Dict[int, Dict[str, Any]]:
    
    """Automatically name each cluster using CLIP similarity.
    
    Args:
        centroids: Dict mapping cluster ID to centroid embedding.
        candidate_labels: List of possible cluster names.
        model: CLIP model.
        processor: CLIP processor.
        device: Device string.
        
    Returns:
        Dict mapping cluster ID to {name, confidence, alternatives}.
    """
    print("Naming clusters using CLIP")
    
    # Encode all candidate labels
    text_embeddings = encode_text_labels(candidate_labels, model, processor, device)
    
    cluster_names = {}
    
    for cluster_id, centroid in centroids.items():
        
        # Compute cosine similarity between centroid and all text labels
        
        similarities = np.dot(text_embeddings, centroid)
        
        # We need to get top 3 matches here
        
        top_indices = np.argsort(similarities)[::-1][:3]
        
        best_idx = top_indices[0]
        best_name = candidate_labels[best_idx]
        confidence = float(similarities[best_idx])
        
        alternatives = [
            {"name": candidate_labels[idx], "score": float(similarities[idx])}
            for idx in top_indices[1:3]
        ]
        
        cluster_names[cluster_id] = {
            "name": best_name,
            "confidence": round(confidence, 4),
            "alternatives": alternatives
        }
        
        print("Cluster {}: '{}' (conf: {:.2f})".format(cluster_id, best_name, confidence))
    
    
    return cluster_names


def save_results(
    labels: np.ndarray,
    cluster_names: Dict[int, Dict[str, Any]],
    filenames: Optional[List[str]]
) -> None:
    """Save clustering results to disk.
    
    Args:
        labels: Array of cluster labels.
        cluster_names: Dict of cluster naming info.
        filenames: List of image filenames.
    """
    print("Saving results")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save cluster labels as numpy array
    labels_path = OUTPUT_DIR / "cluster_labels.npy"
    np.save(labels_path, labels)
    
    print("Saved cluster labels to: {}".format(labels_path))
    
    # Compute cluster statistics
    label_counts = Counter(labels)
    cluster_stats = {}
    
    for cluster_id, count in label_counts.items():
        key = int(cluster_id)
        if key == -1:
            cluster_stats[key] = {
                "name": "Noise (Unclustered)",
                "count": int(count),
                "confidence": 0.0
            }
        else:
            cluster_stats[key] = {
                "name": cluster_names[cluster_id]["name"],
                "count": int(count),
                "confidence": cluster_names[cluster_id]["confidence"],
                "alternatives": cluster_names[cluster_id]["alternatives"]
            }
    
    # Save cluster metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_images": len(labels),
        "n_clusters": len([k for k in label_counts.keys() if k != -1]),
        "n_noise": int(label_counts.get(-1, 0)),
        
        "settings": {
            "min_cluster_size": config.HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": config.HDBSCAN_MIN_SAMPLES,
            "metric": config.HDBSCAN_METRIC,
            "selection_method": config.HDBSCAN_CLUSTER_SELECTION_METHOD
        },
        
        "clusters": cluster_stats
    }
    
    metadata_path = OUTPUT_DIR / "cluster_metadata.json"
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Saved cluster metadata to: {}".format(metadata_path))
    
    # Print summary
    print("Cluster Summary:")
    
    for cluster_id in sorted(cluster_stats.keys()):
        info = cluster_stats[cluster_id]
        
        print("{:3d}: {:30s} ({:,} images, conf: {:.2f})".format(
            cluster_id,
            info["name"][:30],
            info["count"],
            info["confidence"]
        ))


def main() -> None:
    """Main entry point for cluster discovery."""
    
    print("HDBSCAN Cluster Discovery Pipeline")
    print("Timestamp: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("")
    
    # Load data
    embeddings, filenames = load_data()
    if embeddings is None:
        sys.exit(1)
    
    # Run clustering
    labels = run_hdbscan(embeddings)
    
    # Compute centroids
    centroids = compute_cluster_centroids(embeddings, labels)
    
    if len(centroids) == 0:
        print("ERROR: No clusters found.")
        
        sys.exit(1)
    
    # Load CLIP and name clusters
    
    model, processor, device = load_clip_model()
    
    cluster_names = name_clusters(centroids, CANDIDATE_LABELS, model, processor, device)
    
    # Save results
    
    save_results(labels, cluster_names, filenames)
    
    print("")
    print("Cluster discovery complete!")
    print("Results saved to: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
