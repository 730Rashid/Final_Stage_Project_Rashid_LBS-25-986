"""
CrisisMMD Visualisation App.

A humanitarian aid tool for exploring disaster imagery using AI-powered
semantic search and visual similarity. Built with CLIP embeddings and UMAP.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
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


# Path Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "visualisation" / "umap_data.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy"
IMAGE_FOLDER = PROJECT_ROOT / "data" / "processed" / "clean_data"

print("Starting Application...")


# Load Metadata
try:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except FileNotFoundError:
    print("Could not find {}. Run umap_reduction.py first.".format(DATA_PATH))
    exit()


def parse_event(path):
    """Extract event name from folder structure."""
    path = str(path).replace("\\", "/").lower()
    
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
        if key in path:
            return label
    
    return "Unknown Event"


df["event"] = df["path"].apply(parse_event)
df["filename"] = df["path"].apply(lambda p: Path(p).name)
df["hover"] = df.apply(
    lambda r: "<b>{}</b><br>{}".format(r["event"], r["filename"]), 
    axis=1
)
df["original_idx"] = df.index

UNIQUE_EVENTS = sorted(df["event"].unique())
print("Metadata loaded: {} images across {} events".format(len(df), len(UNIQUE_EVENTS)))


# Load CLIP Model
print("Loading CLIP Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Inference Device: {}".format(device))

try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("Loading Image Embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print("Embeddings loaded: {}".format(embeddings.shape))
except Exception as e:
    print("Failed to load AI models: {}".format(e))
    exit()


# Zero Shot Classification
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

print("Precomputing label embeddings...")
label_embeddings = None

try:
    label_inputs = clip_processor(
        text=CLASSIFICATION_LABELS, 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    with torch.no_grad():
        label_features = clip_model.get_text_features(**label_inputs)
    
    if hasattr(label_features, "pooler_output"):
        label_features = label_features.pooler_output
    elif hasattr(label_features, "last_hidden_state"):
        label_features = label_features.last_hidden_state[:, 0, :]
    
    label_features = label_features / label_features.norm(p=2, dim=-1, keepdim=True)
    label_embeddings = label_features.cpu().numpy()
    print("Label embeddings ready")
except Exception as e:
    print("Could not precompute label embeddings: {}".format(e))


# Search Functions

def semantic_search(query, subset_indices=None, top_k=50):
    """Find images matching the text query."""
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output
    elif hasattr(text_features, "last_hidden_state"):
        text_features = text_features.last_hidden_state[:, 0, :]
    
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    text_vector = text_features.cpu().numpy()
    
    if subset_indices is not None and len(subset_indices) > 0:
        subset_embeddings = embeddings[subset_indices]
        similarities = cosine_similarity(text_vector, subset_embeddings)[0]
        local_top_k = min(top_k, len(subset_indices))
        local_top_indices = np.argsort(similarities)[::-1][:local_top_k]
        global_indices = np.array(subset_indices)[local_top_indices]
        return global_indices, similarities[local_top_indices]
    else:
        similarities = cosine_similarity(text_vector, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices, similarities[top_indices]


def visual_search(image_index, subset_indices=None, top_k=50):
    """Find visually similar images."""
    query_vector = embeddings[image_index].reshape(1, -1)
    
    if subset_indices is not None and len(subset_indices) > 0:
        subset_embeddings = embeddings[subset_indices]
        similarities = cosine_similarity(query_vector, subset_embeddings)[0]
        local_top_k = min(top_k, len(subset_indices))
        local_top_indices = np.argsort(similarities)[::-1][:local_top_k]
        global_indices = np.array(subset_indices)[local_top_indices]
        return global_indices, similarities[local_top_indices]
    else:
        similarities = cosine_similarity(query_vector, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices, similarities[top_indices]


def classify_image(image_index, threshold=0.20):
    """Classify image content using zero shot classification."""
    if label_embeddings is None:
        return []
    
    image_vector = embeddings[image_index].reshape(1, -1)
    similarities = cosine_similarity(image_vector, label_embeddings)[0]
    
    results = []
    for i, score in enumerate(similarities):
        if score >= threshold:
            display_name = LABEL_DISPLAY_NAMES[CLASSIFICATION_LABELS[i]]
            results.append((display_name, float(score)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# Flask App with Bootstrap Theme
BOOTSTRAP_ICONS = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
ASSETS_PATH = str(PROJECT_ROOT / "assets")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LITERA, BOOTSTRAP_ICONS],
    assets_folder=ASSETS_PATH,
    suppress_callback_exceptions=True
)
server = app.server


@server.route("/images/<path:p>")
def serve_image(p):
    """Serve images from the data folder."""
    path = IMAGE_FOLDER / p
    if path.exists():
        return send_from_directory(str(path.parent), path.name)
    return "Not found", 404


# Component Builders

def create_badge(label, confidence):
    """Create a classification badge."""
    if confidence >= 0.28:
        colour = "success"
    elif confidence >= 0.24:
        colour = "primary"
    else:
        colour = "secondary"
    
    return dbc.Badge(
        "{} {:.0f}%".format(label, confidence * 100),
        color=colour,
        className="me-1 mb-1",
        style={"fontSize": "10px"}
    )


def create_stat_card(value, label):
    """Create a statistic card."""
    return dbc.Card([
        dbc.CardBody([
            html.Div(value, className="stat-value"),
            html.Div(label, className="stat-label")
        ], className="text-center")
    ], className="h-100")


# Page Layouts

def overview_page():
    """Overview landing page - clean and readable."""
    return html.Div([
        # Hero Section
        html.Div([
            dbc.Container([
                html.H1("Visualising Natural Disaster Image Embeddings", 
                        className="hero-title"),
                html.P("Semantic Search and Visual Clustering for Humanitarian Response",
                       className="hero-subtitle"),
                html.Div([
                    dbc.Badge("Honours Stage Project", color="light", 
                              text_color="dark", className="me-2"),
                    html.Span("Rashid", className="fw-bold", style={"color": "white"}),
                    html.Span(" | Supervised by ", style={"opacity": "0.8", "color": "white"}),
                    html.Span("XinHui Ma", className="fw-bold", style={"color": "white"})
                ], className="mt-3")
            ], fluid=True)
        ], className="hero-section"),
        
        dbc.Container([
            # Stats Row - Compact
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("{:,}".format(len(df)), className="stat-value"),
                        html.Div("Images", className="stat-label")
                    ], className="stat-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div(str(len(UNIQUE_EVENTS)), className="stat-value"),
                        html.Div("Events", className="stat-label")
                    ], className="stat-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div("512", className="stat-value"),
                        html.Div("Dimensions", className="stat-label")
                    ], className="stat-card")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div(str(len(CLASSIFICATION_LABELS)), className="stat-value"),
                        html.Div("Labels", className="stat-label")
                    ], className="stat-card")
                ], md=3),
            ], className="mb-5"),
            
            # Challenge & Solution - Side by Side, No cards
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("The Challenge", className="mb-3"),
                        html.P([
                            "In the aftermath of a disaster, social media is flooded with ",
                            html.Strong("millions of images"),
                            ". This creates a bottleneck—too much ",
                            html.Strong("visual noise"),
                            " for human teams to filter."
                        ]),
                        html.P([
                            "Traditional AI relies on pre-trained labels and ",
                            html.Strong("fails"),
                            " when encountering new crisis types."
                        ], className="mb-0")
                    ], className="info-block challenge")
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H4("The Solution", className="mb-3"),
                        html.P([
                            "This project uses ",
                            html.Strong("CLIP"),
                            " to extract semantic vectors from images, enabling search by ",
                            html.Strong("natural language"),
                            " rather than keywords."
                        ]),
                        html.P([
                            html.Strong("UMAP"),
                            " projects high-dimensional data to reveal how images cluster naturally."
                        ], className="mb-0")
                    ], className="info-block solution")
                ], md=6),
            ], className="mb-5"),
            
            # How It Works - Simple row, no card
            html.H4("How It Works", className="section-header"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-database fs-4")
                        ], className="feature-icon feature-icon-blue"),
                        html.H6("Ingestion", className="fw-semibold"),
                        html.P("Parse and clean 17,463 crisis images", 
                               className="text-muted small mb-0")
                    ], className="feature-block")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-cpu fs-4")
                        ], className="feature-icon feature-icon-green"),
                        html.H6("Vectorisation", className="fw-semibold"),
                        html.P("Generate 512-dim embeddings via CLIP", 
                               className="text-muted small mb-0")
                    ], className="feature-block")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-diagram-3 fs-4")
                        ], className="feature-icon feature-icon-orange"),
                        html.H6("Projection", className="fw-semibold"),
                        html.P("UMAP reduces to 2D for visualisation", 
                               className="text-muted small mb-0")
                    ], className="feature-block")
                ], md=3),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="bi bi-search fs-4")
                        ], className="feature-icon feature-icon-pink"),
                        html.H6("Search", className="fw-semibold"),
                        html.P("Query in plain English, get results", 
                               className="text-muted small mb-0")
                    ], className="feature-block")
                ], md=3),
            ], className="mb-5"),
            
            # Zero-Shot Feature - Simple info block
            html.Div([
                html.H4("Zero-Shot Semantic Search", className="mb-3"),
                html.P([
                    "You can search for complex concepts like ",
                    html.Em("\"people sleeping in tents\""),
                    " or ",
                    html.Em("\"damaged bridge over water\""),
                    "—the system finds matching images ",
                    html.Strong("mathematically"),
                    ", without manual tagging."
                ]),
                html.Div([
                    create_badge(LABEL_DISPLAY_NAMES[label], 0.28) 
                    for label in CLASSIFICATION_LABELS
                ], className="mt-3")
            ], className="info-block feature mb-5"),
            
            # CTA - Simple and clean
            dbc.Row([
                dbc.Col([
                    html.H5("Ready to Explore?", className="mb-2"),
                    html.P("Search the dataset using natural language.", 
                           className="text-muted mb-0")
                ], md=8, className="d-flex flex-column justify-content-center"),
                dbc.Col([
                    dbc.Button("Open Explorer", href="/explorer", color="primary", 
                               size="lg", className="w-100")
                ], md=4)
            ], className="p-4 bg-light rounded-3 mb-4"),
            
            # Footer with Social Links
            html.Footer([
                html.Hr(className="my-4"),
                html.Div([
                    html.P("Natural Disaster Visualisation Tool", className="fw-bold mb-2"),
                    html.P("Built with CLIP, UMAP, and Dash", className="text-muted small mb-3"),
                    html.Div([
                        html.A([
                            html.I(className="bi bi-github me-2"),
                            "GitHub"
                        ], href="https://github.com/730Rashid", target="_blank",
                           className="footer-link me-4"),
                        html.A([
                            html.I(className="bi bi-linkedin me-2"),
                            "LinkedIn"
                        ], href="https://www.linkedin.com/in/rashid-pandor-85537b22b/", 
                           target="_blank", className="footer-link")
                    ], className="mb-3"),
                    html.P("© 2026 Rashid Pandor", className="text-muted small mb-0")
                ], className="text-center py-4")
            ])
        ], fluid=True, className="py-4")
    ])


def explorer_page():
    """Interactive explorer page."""
    return dbc.Container([
        # Search Bar
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Filter by Event", className="small text-muted"),
                        dcc.Dropdown(
                            id="event-filter",
                            options=[{"label": "All Events", "value": "all"}] +
                                    [{"label": e, "value": e} for e in UNIQUE_EVENTS],
                            value="all",
                            clearable=False
                        )
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Semantic Search", className="small text-muted"),
                        dbc.InputGroup([
                            dbc.Input(
                                id="search-input",
                                type="text",
                                placeholder="e.g. flooded street, rescue boat",
                                debounce=True
                            ),
                            dbc.Button("Search", id="search-btn", n_clicks=0, color="primary"),
                            dbc.Button("Clear", id="clear-btn", n_clicks=0, color="secondary", outline=True)
                        ])
                    ], md=6),
                    dbc.Col([
                        dbc.Label("Status", className="small text-muted"),
                        html.Div(id="search-status", className="text-muted pt-2")
                    ], md=3)
                ], align="end")
            ])
        ], className="mb-3"),
        
        # Tip
        dbc.Alert(
            "Click any point on the map to find visually similar images",
            color="info",
            className="py-2 mb-3 text-center"
        ),
        
        dcc.Store(id="clicked-point-store", data=None),
        
        # Main Content
        dbc.Row([
            # Left: UMAP
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("Embedding Space", className="mb-0 d-inline"),
                        html.Span(" (UMAP)", className="text-muted small")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id="umap-graph",
                            style={"height": "65vh"},
                            config={"displaylogo": False}
                        )
                    ], className="p-2")
                ])
            ], md=7),
            
            # Right: Gallery
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6(id="gallery-title", children="Results", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(
                            id="image-grid",
                            style={"height": "65vh", "overflowY": "auto"}
                        )
                    ], className="p-2")
                ])
            ], md=5)
        ])
    ], fluid=True, className="py-3")


# Navigation
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Overview", href="/", className="text-dark")),
        dbc.NavItem(dbc.NavLink("Explorer", href="/explorer", className="text-dark")),
    ],
    brand="CrisisMMD",
    brand_href="/",
    color="light",
    className="border-bottom mb-0"
)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    html.Div(id="page-content")
])


# Callbacks

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page(pathname):
    if pathname == "/explorer":
        return explorer_page()
    return overview_page()


@app.callback(
    Output("clicked-point-store", "data"),
    [Input("umap-graph", "clickData"), Input("clear-btn", "n_clicks")],
    prevent_initial_call=True
)
def handle_click(click_data, clear_clicks):
    triggered_id = ctx.triggered_id
    if triggered_id == "clear-btn":
        return None
    if click_data and "points" in click_data:
        point = click_data["points"][0]
        if "pointIndex" in point:
            return point["pointIndex"]
    return None


def build_image_card(row, score, score_label="Match"):
    """Build an image card with classification badges."""
    try:
        rel_path = Path(row["path"]).relative_to(IMAGE_FOLDER)
        img_url = "/images/{}".format(str(rel_path).replace(os.sep, "/"))
        
        image_idx = row["original_idx"]
        classifications = classify_image(image_idx, threshold=0.22)
        
        badge_elements = [create_badge(label, conf) for label, conf in classifications[:3]]
        
        return dbc.Card([
            dbc.CardImg(src=img_url, top=True, className="gallery-img"),
            dbc.CardBody([
                html.Small(row["event"], className="text-muted d-block"),
                html.Span(
                    "{}: {:.0f}%".format(score_label, score * 100),
                    className="fw-bold text-primary"
                ),
                html.Div(badge_elements, className="mt-2") if badge_elements else None
            ], className="p-2")
        ], className="mb-2")
    except ValueError:
        return None


@app.callback(
    [
        Output("umap-graph", "figure"),
        Output("image-grid", "children"),
        Output("search-status", "children"),
        Output("gallery-title", "children")
    ],
    [
        Input("search-btn", "n_clicks"),
        Input("search-input", "n_submit"),
        Input("event-filter", "value"),
        Input("clicked-point-store", "data")
    ],
    [State("search-input", "value")]
)
def update_view(n_clicks, n_submit, selected_event, clicked_index, query):
    """Update the visualisation based on user interaction."""
    fig = go.Figure()
    images = []
    status = "Ready"
    gallery_title = "Results"
    
    if selected_event and selected_event != "all":
        event_mask = df["event"] == selected_event
        filtered_df = df[event_mask]
        ghosted_df = df[~event_mask]
        filtered_indices = filtered_df["original_idx"].tolist()
        status = "{:,} images in {}".format(len(filtered_df), selected_event)
    else:
        filtered_df = df
        ghosted_df = pd.DataFrame()
        filtered_indices = None
        status = "{:,} images".format(len(df))
    
    # Ghost layer
    if len(ghosted_df) > 0:
        fig.add_trace(go.Scattergl(
            x=ghosted_df["x"],
            y=ghosted_df["y"],
            mode="markers",
            marker=dict(size=4, color="#e0e0e0", opacity=0.2),
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Active layer
    fig.add_trace(go.Scattergl(
        x=filtered_df["x"],
        y=filtered_df["y"],
        mode="markers",
        marker=dict(size=5, color="#6c9bd1", opacity=0.5),
        text=filtered_df["hover"],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    ))

    # Visual Query
    if clicked_index is not None:
        indices, scores = visual_search(clicked_index, subset_indices=filtered_indices)
        match_df = df.iloc[indices].copy()
        match_df["score"] = scores
        
        query_row = df.iloc[clicked_index]
        fig.add_trace(go.Scattergl(
            x=[query_row["x"]],
            y=[query_row["y"]],
            mode="markers",
            marker=dict(
                size=16, color="#e53935", opacity=1.0,
                line=dict(width=2, color="white"), symbol="star"
            ),
            hovertemplate="<b>Query</b><extra></extra>",
            showlegend=False
        ))
        
        fig.add_trace(go.Scattergl(
            x=match_df["x"],
            y=match_df["y"],
            mode="markers",
            marker=dict(
                size=9, color="#27ae60", opacity=0.9,
                line=dict(width=1, color="white")
            ),
            text=match_df["hover"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        ))
        
        status = "{} similar images".format(len(indices))
        gallery_title = "Similar Images"
        
        for _, row in match_df.head(9).iterrows():
            card = build_image_card(row, row["score"], "Similarity")
            if card:
                images.append(dbc.Col(card, width=4))

    # Text Search
    elif query and len(query.strip()) > 2:
        indices, scores = semantic_search(query.strip(), subset_indices=filtered_indices)
        match_df = df.iloc[indices].copy()
        match_df["score"] = scores
        
        # Filter by minimum threshold (28%) to remove weak matches
        MIN_THRESHOLD = 0.28
        strong_matches = match_df[match_df["score"] >= MIN_THRESHOLD]
        
        if len(strong_matches) > 0:
            fig.add_trace(go.Scattergl(
                x=strong_matches["x"],
                y=strong_matches["y"],
                mode="markers",
                marker=dict(
                    size=10, color="#3498db", opacity=1.0,
                    line=dict(width=1, color="white")
                ),
                text=strong_matches["hover"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            ))
            
            status = "{} relevant matches for '{}' (≥28% similarity)".format(len(strong_matches), query)
            gallery_title = "Search Results"
            
            for _, row in strong_matches.head(9).iterrows():
                card = build_image_card(row, row["score"], "Match")
                if card:
                    images.append(dbc.Col(card, width=4))
        else:
            # No strong matches - show message
            status = "No strong matches for '{}' (try a different query)".format(query)
            gallery_title = "No Results"
            images.append(html.Div([
                html.I(className="bi bi-search fs-1 text-muted"),
                html.P("No images matched '{}' with ≥28% similarity.".format(query), 
                       className="text-muted mt-2"),
                html.P("Try more specific terms like 'flooded street' or 'rescue helicopter'.",
                       className="small text-muted")
            ], className="text-center py-5"))

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan"
    )
    
    # Wrap images in a Row for Bootstrap grid
    image_grid = dbc.Row(images, className="g-2") if images else html.Div("Search or click to see results", className="text-muted text-center py-5")
    
    return fig, image_grid, status, gallery_title


if __name__ == "__main__":
    print("\nServer running at http://127.0.0.1:8050/")
    app.run(debug=False)
