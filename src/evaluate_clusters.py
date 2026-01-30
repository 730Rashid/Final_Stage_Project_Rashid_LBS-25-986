"""
Cluster Evaluation Script for UMAP Embeddings.

This script evaluates the quality of the 2D UMAP projection by measuring
how well the ground-truth event categories are separated in the reduced
space. It provides quantitative metrics for the dissertation.

Metrics Calculated:
    - Silhouette Score: Measures cluster cohesion and separation (-1 to 1).
    - Davies-Bouldin Index: Lower values indicate better clustering.
    - Calinski-Harabasz Index: Higher values indicate denser, well-separated clusters.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import LabelEncoder


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Paths
UMAP_COORDS_PATH = PROJECT_ROOT / "data" / "visualisation" / "umap_coords.npy"
FILENAMES_PATH = PROJECT_ROOT / "data" / "embeddings" / "filenames.json"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "metrics"


def parse_event_from_path(filepath: str) -> str:
    """
    Extract the event type from an image file path.
    
    The CrisisMMD dataset has folder names like 'california_wildfires',
    'hurricane_harvey', etc. This function parses those labels.
    
    Args:
        filepath: Full path to the image file.
        
    Returns:
        Human-readable event name (e.g., 'California Wildfires').
    """
    path_str = str(filepath).replace("\\", "/").lower()
    
    # Event mapping based on CrisisMMD folder structure
    event_mappings = {
        "california_wildfires": "California Wildfires",
        "hurricane_harvey": "Hurricane Harvey",
        "hurricane_irma": "Hurricane Irma",
        "hurricane_maria": "Hurricane Maria",
        "iraq_iran_earthquake": "Iraq-Iran Earthquake",
        "mexico_earthquake": "Mexico Earthquake",
        "srilanka_floods": "Sri Lanka Floods",
    }
    
    for key, label in event_mappings.items():
        if key in path_str:
            return label
    
    return "Unknown"


def load_data():
    """Load UMAP coordinates and extract event labels from filenames."""
    print("Loading data...")
    
    # Load 2D coordinates
    if not UMAP_COORDS_PATH.exists():
        print("ERROR: UMAP coordinates not found at {}".format(UMAP_COORDS_PATH))
        print("Run umap_reduction.py first.")
        return None, None
    
    coords = np.load(UMAP_COORDS_PATH)
    print("  Loaded {} 2D coordinates".format(len(coords)))
    
    # Load filenames and extract event labels
    if not FILENAMES_PATH.exists():
        print("ERROR: Filenames not found at {}".format(FILENAMES_PATH))
        return None, None
    
    with open(FILENAMES_PATH, "r") as f:
        filenames = json.load(f)
    
    # Parse event labels
    events = [parse_event_from_path(path) for path in filenames]
    print("  Extracted {} event labels".format(len(events)))
    
    return coords, events


def calculate_metrics(coords: np.ndarray, labels: list) -> dict:
    """
    Calculate clustering quality metrics.
    
    Args:
        coords: 2D UMAP coordinates of shape (N, 2).
        labels: Ground-truth event labels for each point.
        
    Returns:
        Dictionary containing metric values and interpretation.
    """
    print("\nCalculating clustering metrics...")
    
    # Encode string labels to integers for sklearn
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # Calculate metrics
    silhouette = silhouette_score(coords, encoded_labels)
    davies_bouldin = davies_bouldin_score(coords, encoded_labels)
    calinski = calinski_harabasz_score(coords, encoded_labels)
    
    # Interpretation thresholds
    if silhouette > 0.5:
        silhouette_quality = "Strong separation"
    elif silhouette > 0.25:
        silhouette_quality = "Moderate separation"
    elif silhouette > 0:
        silhouette_quality = "Weak separation"
    else:
        silhouette_quality = "Poor (overlapping clusters)"
    
    if davies_bouldin < 1.0:
        db_quality = "Excellent clustering"
    elif davies_bouldin < 2.0:
        db_quality = "Good clustering"
    else:
        db_quality = "Poor clustering"
    
    metrics = {
        "silhouette_score": {
            "value": round(silhouette, 4),
            "range": "[-1, 1] (higher is better)",
            "interpretation": silhouette_quality
        },
        "davies_bouldin_index": {
            "value": round(davies_bouldin, 4),
            "range": "[0, inf) (lower is better)",
            "interpretation": db_quality
        },
        "calinski_harabasz_index": {
            "value": round(calinski, 2),
            "range": "[0, inf) (higher is better)",
            "interpretation": "Relative measure; compare across configurations"
        }
    }
    
    return metrics, encoder.classes_


def generate_report(metrics: dict, events: list, classes: np.ndarray):
    """Generate and save the evaluation report."""
    print("\nGenerating evaluation report...")
    
    # Count samples per class
    class_counts = Counter(events)
    
    # Build report
    report = {
        "title": "UMAP Cluster Evaluation Report",
        "generated_at": datetime.now().isoformat(),
        "dataset": "CrisisMMD",
        "total_samples": len(events),
        "num_classes": len(classes),
        "class_distribution": dict(class_counts),
        "metrics": metrics
    }
    
    # Save as JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "cluster_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  Saved JSON report: {}".format(json_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTER EVALUATION RESULTS")
    print("=" * 60)
    print("Dataset: CrisisMMD ({:,} images)".format(len(events)))
    print("Number of Event Categories: {}".format(len(classes)))
    print("")
    print("Class Distribution:")
    for event, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(events)
        print("  {:25s} {:5,} ({:5.1f}%)".format(event, count, pct))
    print("")
    print("Clustering Metrics:")
    print("-" * 60)
    
    for name, data in metrics.items():
        print("  {}".format(name.replace("_", " ").title()))
        print("    Value:          {}".format(data["value"]))
        print("    Range:          {}".format(data["range"]))
        print("    Interpretation: {}".format(data["interpretation"]))
        print("")
    
    print("=" * 60)
    print("Report saved to: {}".format(json_path))
    
    return report


def main():
    """Main entry point."""
    print("UMAP Cluster Evaluation Pipeline")
    print("Timestamp: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("")
    
    # Load data
    coords, events = load_data()
    if coords is None:
        sys.exit(1)
    
    # Validate data alignment
    if len(coords) != len(events):
        print("ERROR: Mismatch between coordinates ({}) and labels ({})".format(
            len(coords), len(events)
        ))
        sys.exit(1)
    
    # Calculate metrics
    metrics, classes = calculate_metrics(coords, events)
    
    # Generate report
    generate_report(metrics, events, classes)
    
    print("\nEvaluation complete.")
    print("Use these metrics in your dissertation to justify UMAP's effectiveness.")


if __name__ == "__main__":
    main()
