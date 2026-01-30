"""
CLIP Embedding Prototype for CrisisMMD Dataset.

This script demonstrates the core embedding and visualisation pipeline.
It loads a small sample of disaster images, generates CLIP embeddings,
reduces them to 2D using UMAP, and creates an interactive scatter plot.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import torch
import clip
from PIL import Image
import numpy as np
import umap
import plotly.graph_objects as go
from pathlib import Path
import random
import webbrowser


# Configuration
DATASET_PATH = r"Prototype/data_image"
NUM_IMAGES = 20
OUTPUT_FILE = "prototype.html"


def find_images(data_path):
    """
    Find all image files in the given directory.
    
    Args:
        data_path: Path object pointing to the image directory.
        
    Returns:
        List of Path objects for each image found.
    """
    print("Finding images...")
    
    jpg_images = list(data_path.rglob("*.jpg"))[:50]
    png_images = list(data_path.rglob("*.png"))[:50]
    all_images = jpg_images + png_images
    
    if not all_images:
        print("No images found. Check your path: {}".format(DATASET_PATH))
        return []
    
    print("Found {} jpg and {} png images".format(len(jpg_images), len(png_images)))
    return all_images


def load_clip_model():
    """
    Load the CLIP model.
    
    Returns:
        tuple: (model, preprocess function, device string)
    """
    print("Loading CLIP model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    print("Using device: {}".format(device))
    return model, preprocess, device


def generate_embeddings(model, preprocess, device, images, events):
    """
    Generate CLIP embeddings for a list of images.
    
    Args:
        model: The CLIP model.
        preprocess: The preprocessing function.
        device: The device to run on.
        images: List of image paths.
        events: List of event labels for each image.
        
    Returns:
        tuple: (embeddings array, valid image paths, valid events)
    """
    print("Generating embeddings...")
    
    embeddings = []
    valid_images = []
    valid_events = []
    
    for img_path, event in zip(images, events):
        try:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model.encode_image(image_input)
                embeddings.append(embedding.cpu().numpy().flatten())
                valid_images.append(img_path)
                valid_events.append(event)
                print("  Processed: {}".format(img_path.name))
                
        except Exception as e:
            print("  Skipped {}: {}".format(img_path.name, e))
    
    return np.array(embeddings), valid_images, valid_events


def reduce_dimensions(embeddings):
    """
    Reduce embeddings to 2D using UMAP.
    
    Args:
        embeddings: Array of shape (n_samples, 512).
        
    Returns:
        Array of shape (n_samples, 2).
    """
    print("Reducing to 2D with UMAP...")
    
    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    coords = reducer.fit_transform(embeddings)
    
    print("Dimensionality reduction complete")
    return coords


def create_visualisation(coords, valid_images, valid_events):
    """
    Create an interactive scatter plot of the embeddings.
    
    Args:
        coords: 2D coordinates from UMAP.
        valid_images: List of image paths.
        valid_events: List of event labels.
        
    Returns:
        Plotly figure object.
    """
    print("Creating visualisation...")
    
    # Create colour mapping
    unique_events = list(set(valid_events))
    colour_map = {event: i for i, event in enumerate(unique_events)}
    colours = [colour_map[event] for event in valid_events]
    
    # Build the figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(
            size=15,
            color=colours,
            colorscale="Viridis",
            showscale=True,
            line=dict(width=2, color="white")
        ),
        text=["{}  {}".format(event, path.name) 
              for path, event in zip(valid_images, valid_events)],
        hovertemplate="%{text}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Disaster Image Embeddings Prototype",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        width=1000,
        height=700,
        plot_bgcolor="white"
    )
    
    return fig


def main():
    """
    Main entry point for the prototype.
    """
    print("Starting CrisisMMD Embedding Prototype...")
    
    # Find images
    data_path = Path(DATASET_PATH)
    all_images = find_images(data_path)
    
    if not all_images:
        return
    
    # Sample a subset
    images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
    print("Selected {} images for processing".format(len(images)))
    
    # Extract event labels from folder names
    events = []
    for img in images:
        relative = str(img.relative_to(data_path))
        # Handle both Windows and Unix path separators
        parts = relative.replace("\\", "/").split("/")
        events.append(parts[0])
    
    print("Events found: {}".format(set(events)))
    print("")
    
    # Load model
    model, preprocess, device = load_clip_model()
    print("")
    
    # Generate embeddings
    embeddings, valid_images, valid_events = generate_embeddings(
        model, preprocess, device, images, events
    )
    print("Total embeddings: {}".format(len(embeddings)))
    print("")
    
    # Reduce dimensions
    coords = reduce_dimensions(embeddings)
    print("")
    
    # Create visualisation
    fig = create_visualisation(coords, valid_images, valid_events)
    fig.write_html(OUTPUT_FILE)
    print("Visualisation saved to: {}".format(OUTPUT_FILE))
    
    # Open in browser
    print("Opening in browser...")
    webbrowser.open("file://{}".format(Path(OUTPUT_FILE).absolute()))
    
    print("")
    print("Prototype complete")
    print("")


if __name__ == "__main__":
    main()