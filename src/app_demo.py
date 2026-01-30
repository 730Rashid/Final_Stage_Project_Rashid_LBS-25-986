"""
CrisisMMD Visualization App - Production Edition.
Real-time CLIP semantic search with UMAP visualization.

Features:
    - Event Type Filtering: Filter by disaster event before searching.
    - Semantic Search: Natural language queries using CLIP.
    - UMAP Visualization: Interactive 2D embedding map.
    - Ghost Mode: Non-selected events are faded for context.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from flask import send_from_directory
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- PATH CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "visualisation" / "umap_data.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy"
IMAGE_FOLDER = PROJECT_ROOT / "data" / "processed" / "clean_data"

print("Starting this Application")

# --- 1. LOAD METADATA ---
try:
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except FileNotFoundError:
    print(f"[Error] Could not find {DATA_PATH}. Run umap_reduction.py first.")
    exit()


def parse_event(path):
    """Extracts event name from folder structure for cleaner tooltips."""
    path = str(path).replace("\\", "/")
    if "california_wildfires" in path.lower(): return "California Wildfires"
    if "hurricane_harvey" in path.lower(): return "Hurricane Harvey"
    if "hurricane_irma" in path.lower(): return "Hurricane Irma"
    if "hurricane_maria" in path.lower(): return "Hurricane Maria"
    if "iraq_iran_earthquake" in path.lower(): return "Iraq-Iran Earthquake"
    if "mexico_earthquake" in path.lower(): return "Mexico Earthquake"
    if "srilanka_floods" in path.lower(): return "Sri Lanka Floods"
    return "Unknown Event"


df["event"] = df["path"].apply(parse_event)
df["filename"] = df["path"].apply(lambda p: Path(p).name)
df["hover"] = df.apply(lambda r: f"<b>{r['event']}</b><br>{r['filename']}", axis=1)

# Store original index for embedding lookup
df["original_idx"] = df.index

# Get unique events for dropdown
UNIQUE_EVENTS = sorted(df["event"].unique())
print(f"[Status] Metadata loaded: {len(df)} images across {len(UNIQUE_EVENTS)} events")


# --- 2. GLOBAL MODEL LOADING (Optimisation) ---
print("[Status] Loading CLIP Model (ViT-B/32)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Status] Inference Device: {device}")

try:
    # Use Hugging Face Transformers to match vectorise.py
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("[Status] Loading Image Embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"[Status] Embeddings loaded. Shape: {embeddings.shape}")
except Exception as e:
    print(f"[Error] Failed to load AI models: {e}")
    exit()


# --- 3. SEMANTIC SEARCH ENGINE ---
def semantic_search(query, subset_indices=None, top_k=50):
    """
    Encode query and find most similar images.
    
    Args:
        query: Natural language search query.
        subset_indices: Optional list of indices to search within.
                       If None, searches all embeddings.
        top_k: Number of results to return.
        
    Returns:
        Tuple of (indices into df, similarity scores).
    """
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    
    # Normalise
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    text_vector = text_features.cpu().numpy()
    
    # Search within subset or full dataset
    if subset_indices is not None and len(subset_indices) > 0:
        # Search only within the filtered subset
        subset_embeddings = embeddings[subset_indices]
        similarities = cosine_similarity(text_vector, subset_embeddings)[0]
        
        # Get top K within subset
        local_top_k = min(top_k, len(subset_indices))
        local_top_indices = np.argsort(similarities)[::-1][:local_top_k]
        
        # Map back to original indices
        global_indices = np.array(subset_indices)[local_top_indices]
        top_scores = similarities[local_top_indices]
        
        return global_indices, top_scores
    else:
        # Search all embeddings
        similarities = cosine_similarity(text_vector, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices, similarities[top_indices]


# --- 4. FLASK SERVER ---
# Create a single Dash app instance (fixes blueprint conflict error)
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


@server.route("/images/<path:p>")
def serve_image(p):
    """Serves images directly from the data folder."""
    path = IMAGE_FOLDER / p
    if path.exists():
        return send_from_directory(str(path.parent), path.name)
    return "Not found", 404


# --- 5. STYLES ---
NAV_STYLE = {
    "display": "flex", "borderBottom": "1px solid #ddd", "padding": "15px 40px",
    "backgroundColor": "white", "alignItems": "center", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"
}
LINK_STYLE = {
    "marginRight": "30px", "textDecoration": "none", "color": "#005a8c",
    "fontWeight": "bold", "fontSize": "16px", "fontFamily": "Segoe UI"
}
DROPDOWN_STYLE = {
    "width": "200px", "marginRight": "20px", "display": "inline-block", "verticalAlign": "middle"
}


# --- 6. LAYOUTS ---
def overview_page():
    """Create the overview/landing page."""
    return html.Div([
        html.Div([
            html.H1("Unsupervised Multimodal Exploration", style={"color": "#005a8c", "fontWeight": "bold"}),
            html.H3("Honours Stage Project | Midway Technical Review", style={"color": "#555", "fontWeight": "normal"}),
            html.P([
                html.Span("Student: ", style={"fontWeight": "bold"}), "Rashid  |  ",
                html.Span("Dataset: ", style={"fontWeight": "bold"}), "CrisisMMD (17,463 Images)"
            ], style={"color": "#666", "fontSize": "18px"})
        ], style={"borderBottom": "1px solid #eee", "paddingBottom": "30px", "marginBottom": "40px"}),

        html.Div([
            html.H4("System Architecture", style={"color": "#333"}),
            html.Div([
                html.Strong("1. Ingestion"), " -> ",
                html.Strong("2. Vectorisation (CLIP)"), " -> ",
                html.Strong("3. Projection (UMAP)"), " -> ",
                html.Strong("4. Interface (Dash)", style={"color": "#005a8c"})
            ], style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "8px", "textAlign": "center"})
        ])
    ], style={"padding": "60px", "maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Segoe UI"})


def explorer_page():
    """Create the interactive explorer page with event filtering."""
    return html.Div([
        # Filter & Search Bar
        html.Div([
            # Event Filter Dropdown
            html.Div([
                html.Label("Filter by Event:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="event-filter",
                    options=[{"label": "All Events", "value": "all"}] + 
                            [{"label": e, "value": e} for e in UNIQUE_EVENTS],
                    value="all",
                    clearable=False,
                    style={"width": "200px", "display": "inline-block", "verticalAlign": "middle"}
                )
            ], style={"display": "inline-block", "marginRight": "30px"}),
            
            # Separator
            html.Span("|", style={"color": "#ccc", "marginRight": "30px", "fontSize": "24px"}),
            
            # Search Input
            html.Label("Semantic Search:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Input(
                id="search-input", type="text", placeholder='e.g., "flooded house"', debounce=True,
                style={"padding": "10px", "width": "300px", "borderRadius": "4px", "border": "1px solid #ccc"}
            ),
            html.Button("Search", id="search-btn", n_clicks=0, style={
                "padding": "10px 25px", "marginLeft": "10px", "backgroundColor": "#005a8c",
                "color": "white", "border": "none", "borderRadius": "4px", "cursor": "pointer"
            }),
            html.Div(id="search-status", style={"display": "inline-block", "marginLeft": "15px", "color": "#666"})
        ], style={"padding": "20px", "backgroundColor": "#f8f9fa", "borderBottom": "1px solid #ddd", "textAlign": "center"}),

        # Main Content
        html.Div([
            # Left: Map
            html.Div([
                html.H4("Global Structure (UMAP)", style={"textAlign": "center", "color": "#555"}),
                dcc.Graph(id="umap-graph", style={"height": "75vh"}, config={"displaylogo": False})
            ], style={"width": "58%", "display": "inline-block", "verticalAlign": "top"}),

            # Right: Gallery
            html.Div([
                html.H4("Visual Evidence (Top 9)", style={"textAlign": "center", "color": "#555"}),
                html.Div(id="image-grid", style={
                    "display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px",
                    "padding": "10px", "height": "75vh", "overflowY": "auto"
                })
            ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"})
        ], style={"padding": "20px"})
    ], style={"fontFamily": "Segoe UI"})


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div([
        html.H2("CrisisMMD Analysis", style={"marginRight": "50px", "color": "#333"}),
        dcc.Link("Overview", href="/", style=LINK_STYLE),
        dcc.Link("Live System", href="/explorer", style=LINK_STYLE),
    ], style=NAV_STYLE),
    html.Div(id="page-content")
])


# --- 7. CALLBACKS ---
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    """Route to the correct page."""
    return explorer_page() if pathname == "/explorer" else overview_page()


@app.callback(
    [Output("umap-graph", "figure"), Output("image-grid", "children"), Output("search-status", "children")],
    [Input("search-btn", "n_clicks"), Input("event-filter", "value")],
    [State("search-input", "value")]
)
def update_view(n_clicks, selected_event, query):
    """
    Update the UMAP plot and image gallery based on event filter AND search.
    
    Logic:
        1. If event is selected: Ghost (grey out) all other events.
        2. If query is entered: Search within the filtered subset.
        3. Both filters work with AND logic.
    """
    fig = go.Figure()
    images = []
    status = "Ready"
    
    # Determine which points belong to the selected event
    if selected_event and selected_event != "all":
        # Filter mode: show selected event highlighted, others ghosted
        event_mask = df["event"] == selected_event
        filtered_df = df[event_mask]
        ghosted_df = df[~event_mask]
        filtered_indices = filtered_df["original_idx"].tolist()
        status = f"Showing {len(filtered_df):,} images from {selected_event}"
    else:
        # No filter: all points are in the active set
        filtered_df = df
        ghosted_df = pd.DataFrame()  # Empty
        filtered_indices = None
        status = f"Showing all {len(df):,} images"
    
    # A. Ghost Layer (Non-selected events - very faded)
    if len(ghosted_df) > 0:
        fig.add_trace(go.Scattergl(
            x=ghosted_df["x"], y=ghosted_df["y"], mode="markers",
            marker=dict(size=4, color="#d0d0d0", opacity=0.15),
            hoverinfo="skip", showlegend=False, name="Other Events"
        ))
    
    # B. Active Event Layer (Selected event or all - slightly visible)
    fig.add_trace(go.Scattergl(
        x=filtered_df["x"], y=filtered_df["y"], mode="markers",
        marker=dict(size=5, color="#a0c4e8", opacity=0.4),
        text=filtered_df["hover"], hovertemplate="%{text}<extra></extra>",
        showlegend=False, name="Active"
    ))

    # C. Search Logic (AND with event filter)
    if query and len(query.strip()) > 2:
        indices, scores = semantic_search(query.strip(), subset_indices=filtered_indices)
        match_df = df.iloc[indices].copy()
        match_df["score"] = scores
        
        # Matches Layer (Blue highlights)
        fig.add_trace(go.Scattergl(
            x=match_df["x"], y=match_df["y"], mode="markers",
            marker=dict(size=12, color="#1976D2", opacity=1.0, line=dict(width=1.5, color="white")),
            text=match_df["hover"], hovertemplate="%{text}<extra></extra>",
            name="Matches", showlegend=False
        ))
        
        if selected_event and selected_event != "all":
            status = f"Found {len(indices)} matches for '{query}' in {selected_event}"
        else:
            status = f"Found {len(indices)} matches for '{query}'"
        
        # Build Gallery (Top 9)
        for _, row in match_df.head(9).iterrows():
            try:
                # Calculate relative path for Flask
                rel_path = Path(row["path"]).relative_to(IMAGE_FOLDER)
                img_url = f"/images/{str(rel_path).replace(os.sep, '/')}"
                
                images.append(html.Div([
                    html.Img(src=img_url, style={"width": "100%", "borderRadius": "4px", "border": "1px solid #ddd"}),
                    html.Div([
                        html.Div(f"{row['event']}", style={"fontSize": "10px", "color": "#888"}),
                        html.Div(f"Score: {row['score']:.2f}", style={"fontSize": "11px", "color": "#005a8c", "fontWeight": "bold"})
                    ], style={"textAlign": "center"})
                ]))
            except ValueError:
                continue

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False), dragmode="pan"
    )
    
    return fig, images, status


if __name__ == "__main__":
    print(f"\n Server running at http://127.0.0.1:8050/")
    app.run(debug=False)
