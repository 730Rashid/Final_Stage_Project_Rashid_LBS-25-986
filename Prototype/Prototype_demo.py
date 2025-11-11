<<<<<<< HEAD
import torch
import clip
from PIL import Image
import numpy as np
import umap
import plotly.graph_objects as go
from pathlib import Path
import random


CRISISMD_PATH = r"data_image"  
NUM_IMAGES = 20  


print(" Starting simple prototype...\n")


print(" Finding images...")
data_path = Path(CRISISMD_PATH)


jpg_images = list(data_path.rglob("*.jpg"))[:50]
png_images = list(data_path.rglob("*.png"))[:50]
all_images = jpg_images + png_images

if not all_images:
    print(f" No images found! Check your path: {CRISISMD_PATH}")
    print(f"   Looking for .jpg or .png files")
    exit()

print(f"   Found {len(jpg_images)} .jpg and {len(png_images)} .png images")


images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
print(f"   Found {len(images)} images")


events = [str(img.relative_to(data_path)).split('\\')[0] for img in images]
print(f"   Events: {set(events)}")


print("\n Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"   Using: {device}")

print("\n Generating embeddings...")
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
            print(f" Working: {img_path.name}")
            
    except Exception as e:
        print(f"  Something went wrong with: {img_path.name}: {e}")

embeddings = np.array(embeddings)
print(f"\n   Total embeddings: {len(embeddings)}")


print("\n Reducing to 2D with UMAP...")
reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, random_state=42)
coords_2d = reducer.fit_transform(embeddings)
print("Done")


print("\n Creating visualisation...")


unique_events = list(set(valid_events))
color_map = {event: i for i, event in enumerate(unique_events)}
colors = [color_map[event] for event in valid_events]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=coords_2d[:, 0],
    y=coords_2d[:, 1],
    mode='markers',
    marker=dict(
        size=15,
        color=colors,
        colorscale='Viridis',
        showscale=True,
        line=dict(width=2, color='white')
    ),
    text=[f"{event}<br>{path.name}" for path, event in zip(valid_images, valid_events)],
    hovertemplate='<b>%{text}</b><extra></extra>'
))

fig.update_layout(
    title="Disaster Image Embeddings Simple Prototype",
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2",
    width=1000,
    height=700,
    plot_bgcolor='white'
)

output_file = "simple_prototype.html"
fig.write_html(output_file)

print(f"\n Comepleted! Visualisation saved to: {output_file}")
print(f"   Opening in browser...")



import webbrowser

webbrowser.open(f"file://{Path(output_file).absolute()}")

print("\n Prototype complete!")
=======
import torch
import clip
from PIL import Image
import numpy as np
import umap
import plotly.graph_objects as go
from pathlib import Path
import random


CRISISMD_PATH = r"data_image"  
NUM_IMAGES = 20  


print(" Starting simple prototype...\n")


print(" Finding images...")
data_path = Path(CRISISMD_PATH)


jpg_images = list(data_path.rglob("*.jpg"))[:50]
png_images = list(data_path.rglob("*.png"))[:50]
all_images = jpg_images + png_images

if not all_images:
    print(f" No images found! Check your path: {CRISISMD_PATH}")
    print(f"   Looking for .jpg or .png files")
    exit()

print(f"   Found {len(jpg_images)} .jpg and {len(png_images)} .png images")


images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
print(f"   Found {len(images)} images")


events = [str(img.relative_to(data_path)).split('\\')[0] for img in images]
print(f"   Events: {set(events)}")


print("\n Loading CLIP model...")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
print(f"   Using: {device}")

print("\n Generating embeddings...")
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
            print(f"  Working: {img_path.name}")
            
    except Exception as e:
        print(f"  Something went wrong with: {img_path.name}: {e}")

embeddings = np.array(embeddings)
print(f"\n   Total embeddings: {len(embeddings)}")


print("\n Reducing to 2D with UMAP...")
reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, random_state=42)
coords_2d = reducer.fit_transform(embeddings)
print("Done")


print("\n Creating visualisation...")


unique_events = list(set(valid_events))
color_map = {event: i for i, event in enumerate(unique_events)}
colors = [color_map[event] for event in valid_events]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=coords_2d[:, 0],
    y=coords_2d[:, 1],
    mode='markers',
    marker=dict(
        size=15,
        color=colors,
        colorscale='Viridis',
        showscale=True,
        line=dict(width=2, color='white')
    ),
    text=[f"{event}<br>{path.name}" for path, event in zip(valid_images, valid_events)],
    hovertemplate='<b>%{text}</b><extra></extra>'
))

fig.update_layout(
    title="Disaster Image Embeddings Simple Prototype",
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2",
    width=1000,
    height=700,
    plot_bgcolor='white'
)

output_file = "simple_prototype.html"
fig.write_html(output_file)

print(f"\n Comepleted! Visualisation saved to: {output_file}")
print(f"   Opening in browser...")



import webbrowser

webbrowser.open(f"file://{Path(output_file).absolute()}")

print("\n Prototype complete!")
>>>>>>> 13a026cf1ef788a70351bb33fa62d3586623e8e1
print("You can hover over points to see image names and events")