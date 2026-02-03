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


# Flask App
BOOTSTRAP_ICONS = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
ASSETS_PATH = str(PROJECT_ROOT / "assets")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, BOOTSTRAP_ICONS],
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
    """Create a classification badge using Academic colours."""
    # Confidence dictates the 'strength' of the evidence
    
    score_pct = confidence * 100
    
    badge_class = "badge-academic"
    if confidence >= 0.28:
        badge_class += " badge-high"
    elif confidence >= 0.24:
        badge_class += " badge-mid"

    # Using standard academic terms
    return html.Span(
        "{} ({:.0f}%)".format(label, score_pct),
        className=badge_class + " me-1 mb-1"
    )


# Page Layouts

def overview_page():
    """Overview landing page - Research Abstract Style."""
    return html.Div([
        # Hero Wrapper
        html.Div([
            dbc.Container([
                html.Small("HONOURS STAGE PROJECT | 2026", className="hero-supertext"),
                html.H1("Visualising Natural Disaster Image Embeddings", className="mb-3"),
                
                html.P("A Semantic Search and Visual Clustering approach for Humanitarian Response",
                       className="hero-subtitle mb-4"),
                
                html.Div([
                    html.Span("Authored by ", className="text-secondary"),
                    html.Strong("Rashid Pandor", className="text-dark me-3"),
                    html.Span("Supervised by ", className="text-secondary"),
                    html.Strong("XinHui Ma", className="text-dark")
                ], className="mb-4 small"),

                html.Div([
                    dbc.Button("View Data Explorer", href="/explorer", className="btn-primary-action me-3"),
                    dbc.Button("Read Documentation", href="https://github.com/730Rashid", target="_blank", 
                               className="btn-academic")
                ], className="d-flex text-start gap-2")
            ], fluid=True, style={"maxWidth": "1000px"})
        ], className="hero-wrapper"),
        
        dbc.Container([
            # Abstract / Stats
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Key Metrics", className="paper-title border-bottom pb-2 mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Div("{:,}".format(len(df)), className="stat-value"),
                                    html.Div("Total Images Processed", className="stat-label")
                                ], className="stat-box")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(str(len(UNIQUE_EVENTS)), className="stat-value"),
                                    html.Div("Disaster Events", className="stat-label")
                                ], className="stat-box")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div("512", className="stat-value"),
                                    html.Div("Vector Dimensions", className="stat-label")
                                ], className="stat-box")
                            ], width=3),
                        ])
                    ], className="paper-card h-100")
                ], md=12, className="mb-4"),
            ]),
            
            # Methodology Section
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Methodology", className="paper-title border-bottom pb-2 mb-3"),
                        html.P([
                            html.Strong("1. Data Ingestion: "), 
                            "The system processes over 17,000 images from the CrisisMMD dataset, covering seven major natural disaster events."
                        ]),
                        html.P([
                            html.Strong("2. Vectorisation: "), 
                            "We utilise the OpenAI CLIP model (ViT-B/32) to generate 512-dimensional semantic embeddings for each image."
                        ]),
                        html.P([
                            html.Strong("3. Dimensionality Reduction: "), 
                            "Uniform Manifold Approximation and Projection (UMAP) reduces these high-dimensional vectors to 2D for visualisation."
                        ])
                    ], className="paper-card h-100")
                ], md=6),
                
                dbc.Col([
                    html.Div([
                        html.H4("Capabilities", className="paper-title border-bottom pb-2 mb-3"),
                        html.Ul([
                            html.Li("Zero-Shot Classification: Categorising images without explicit training labels."),
                            html.Li("Semantic Search: Retrieving images using natural language queries (e.g. \"flood water rising\")."),
                            html.Li("Visual Similarity: Identifying related imagery based on visual content alone.")
                        ], className="text-secondary small ps-3"),
                        
                        dbc.Alert([
                            html.I(className="bi bi-info-circle me-2"),
                            "This system is designed to aid humanitarian response by filtering visual noise during crises."
                        ], color="light", className="mt-4 small")
                    ], className="paper-card h-100")
                ], md=6),
            ], className="g-4 mb-5"),
            
            # Footer
            html.Div([
                html.Hr(className="text-secondary opacity-25"),
                dbc.Row([
                    dbc.Col([
                        html.P("University Honours Project • 2026", className="footer-text")
                    ], width=6),
                    dbc.Col([
                        html.P("LBS-25-986", className="footer-text text-end")
                    ], width=6)
                ])
            ], className="py-4")
            
        ], fluid=True, style={"maxWidth": "1000px"})
    ])


def explorer_page():
    """Interactive Data Explorer."""
    return dbc.Container([
        # Filter Bar
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Filter by Event", className="small text-secondary fw-bold"),
                    dcc.Dropdown(
                        id="event-filter",
                        options=[{"label": "All Events", "value": "all"}] +
                                [{"label": e, "value": e} for e in UNIQUE_EVENTS],
                        value="all",
                        clearable=False,
                        className="dash-dropdown"
                    )
                ], md=3),
                dbc.Col([
                    dbc.Label("Semantic Query", className="small text-secondary fw-bold"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="search-input",
                            type="text",
                            placeholder="e.g. damaged bridge, temporary shelter...",
                            debounce=True,
                            className="custom-input",
                            style={"borderRight": "none"}
                        ),
                        dbc.Button("Search", id="search-btn", n_clicks=0, className="btn-primary-action"),
                        dbc.Button("Clear", id="clear-btn", n_clicks=0, color="link", className="text-secondary")
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Label("System Status", className="small text-secondary fw-bold"),
                    html.Div(id="search-status", className="text-secondary small pt-2")
                ], md=3)
            ], align="start", className="g-4")
        ], className="paper-card mb-4 py-3"),
        
        dcc.Store(id="clicked-point-store", data=None),
        
        # Main Content
        dbc.Row([
            # Left: UMAP
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5("Data Projection (UMAP)", className="paper-title mb-0"),
                    ], className="paper-header"),
                    
                    dcc.Graph(
                        id="umap-graph",
                        style={"height": "75vh"},
                        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
                    )
                ], className="paper-card p-0") # p-0 for graph to fill
            ], md=8),
            
            # Right: Gallery
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H5(id="gallery-title", children="Selected Results", className="paper-title mb-0"),
                    ], className="paper-header mx-3 mt-3"),
                    
                    html.Div(
                        id="image-grid",
                        style={"height": "75vh", "overflowY": "auto", "padding": "0 16px 16px 16px"}
                    )
                ], className="paper-card p-0")
            ], md=4)
        ], className="g-4")
    ], fluid=True, style={"maxWidth": "1400px"}, className="py-4")


# Navigation
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Project Abstract", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Data Explorer", href="/explorer", active="exact")),
    ],
    brand="CrisisMMD Visualisation",
    brand_href="/",
    color="white",
    className="navbar-custom"
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
            dbc.CardImg(src=img_url, top=True, className="gallery-img", style={"borderRadius": "4px 4px 0 0"}),
            dbc.CardBody([
                html.Div([
                    html.Span(row["event"], className="small text-muted d-block text-truncate"),
                    html.Div([
                        html.Span("{:.0f}%".format(score * 100), className="fw-bold text-dark"),
                        html.Span(" " + score_label.lower(), className="small text-secondary")
                    ])
                ], className="mb-2"),
                html.Div(badge_elements, className="d-flex flex-wrap") if badge_elements else None
            ], className="p-2")
        ], className="mb-3 border bg-white shadow-sm")
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
    status = "Ready to search."
    gallery_title = "Selected Images"
    
    if selected_event and selected_event != "all":
        event_mask = df["event"] == selected_event
        filtered_df = df[event_mask]
        ghosted_df = df[~event_mask]
        filtered_indices = filtered_df["original_idx"].tolist()
        status = "Filtering for {} ({} images)".format(selected_event, len(filtered_df))
    else:
        filtered_df = df
        ghosted_df = pd.DataFrame()
        filtered_indices = None
        status = "Displaying complete dataset ({:,} images)".format(len(df))
    
    # Theme Colors - Academic Light
    COLOR_BG = "rgba(0,0,0,0)" 
    
    # Ghost layer (Background context)
    if len(ghosted_df) > 0:
        fig.add_trace(go.Scattergl(
            x=ghosted_df["x"],
            y=ghosted_df["y"],
            mode="markers",
            marker=dict(size=3, color="#f1f5f9", opacity=0.8), # Very light grey
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Active layer (Primary Data)
    fig.add_trace(go.Scattergl(
        x=filtered_df["x"],
        y=filtered_df["y"],
        mode="markers",
        marker=dict(size=4, color="#94a3b8", opacity=0.6, line=dict(width=0)), # Slate grey
        text=filtered_df["hover"],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    ))

    # Visual Query Selection
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
                size=14, color="#ea580c", opacity=1.0, # Orange highlight
                line=dict(width=2, color="white")
            ),
            hovertemplate="<b>Selected Point</b><extra></extra>",
            showlegend=False
        ))
        
        fig.add_trace(go.Scattergl(
            x=match_df["x"],
            y=match_df["y"],
            mode="markers",
            marker=dict(
                size=6, color="#2563eb", opacity=0.8, # Royal Blue
                line=dict(width=0)
            ),
            text=match_df["hover"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        ))
        
        status = "Identified {} visually similar images.".format(len(indices))
        gallery_title = "Visually Similar Images"
        
        for _, row in match_df.head(10).iterrows():
            card = build_image_card(row, row["score"], "Similarity")
            if card:
                images.append(card)

    # Text Search 
    elif query and len(query.strip()) > 2:
        indices, scores = semantic_search(query.strip(), subset_indices=filtered_indices)
        match_df = df.iloc[indices].copy()
        match_df["score"] = scores
        
        MIN_THRESHOLD = 0.28
        strong_matches = match_df[match_df["score"] >= MIN_THRESHOLD]
        
        if len(strong_matches) > 0:
            fig.add_trace(go.Scattergl(
                x=strong_matches["x"],
                y=strong_matches["y"],
                mode="markers",
                marker=dict(
                    size=8, color="#2563eb", opacity=0.9, # Royal Blue
                    line=dict(width=1, color="white")
                ),
                text=strong_matches["hover"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            ))
            
            status = "Found {} matches for '{}'.".format(len(strong_matches), query)
            gallery_title = "Search Results: {}".format(query)
            
            for _, row in strong_matches.head(10).iterrows():
                card = build_image_card(row, row["score"], "Match")
                if card:
                    images.append(card)
        else:
            status = "No significant matches found for '{}'.".format(query)
            gallery_title = "No Results"
            images.append(html.Div([
                html.P("No images matched your query with sufficient confidence.", 
                       className="text-secondary"),
                html.Small("Try using broader terms like 'flooding' or 'rubble'.", className="text-muted")
            ], className="text-center py-5"))

    fig.update_layout(
        plot_bgcolor="#ffffff", # Explicit white background for academic chart look
        paper_bgcolor="#ffffff",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        dragmode="pan",
        showlegend=False
    )
    
    final_grid = []
    if images and isinstance(images[0], html.Div):
         final_grid = images
    else:
        final_grid = [dbc.Col(img, width=12) for img in images]

    image_grid = dbc.Row(final_grid, className="g-3")
    
    if not images and not query and clicked_index is None:
         image_grid = html.Div([
             html.P("Select an event or click a data point on the projection to view details.", className="text-muted small text-center mt-5")
         ])

    return fig, image_grid, status, gallery_title


if __name__ == "__main__":
    print("\nServer running at http://127.0.0.1:8050/")
    app.run(debug=False)
