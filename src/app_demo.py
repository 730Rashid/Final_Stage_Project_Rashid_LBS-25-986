"""
CrisisMMD Visualisation App - Frontend.

A humanitarian aid tool for exploring disaster imagery using AI-powered
semantic search and visual similarity. Built with CLIP embeddings and UMAP.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import sys
import json
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import cv2
import hashlib
import io
import os
from pathlib import Path
from flask import send_from_directory, request, Response

# Add project root to path for config imports
PROJECT_ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))
from config.settings import config

# Import backend functions
from app_backend import (
    get_manager,
    get_dataframe,
    get_unique_events,
    get_analytics,
    semantic_search,
    visual_search,
    classify_image,
    caption_image_by_index,
    CLASSIFICATION_LABELS,
    LABEL_DISPLAY_NAMES,
    PROJECT_ROOT,
    IMAGE_FOLDER
)


# Initialise backend and get data
manager = get_manager()
df = get_dataframe()
UNIQUE_EVENTS = get_unique_events()


# Load cluster data if available
CLUSTER_LABELS_PATH = PROJECT_ROOT / "data" / "visualisation" / "cluster_labels.npy"
CLUSTER_METADATA_PATH = PROJECT_ROOT / "data" / "visualisation" / "cluster_metadata.json"

cluster_labels = None
cluster_metadata = None

if CLUSTER_LABELS_PATH.exists() and CLUSTER_METADATA_PATH.exists():
    try:
        cluster_labels = np.load(CLUSTER_LABELS_PATH)
        with open(CLUSTER_METADATA_PATH, "r") as f:
            cluster_metadata = json.load(f)
        df["cluster_id"] = cluster_labels
        print("Loaded cluster data: {} clusters".format(cluster_metadata["n_clusters"]))
    except Exception as e:
        print("Warning: Could not load cluster data: {}".format(e))
        cluster_labels = None
        cluster_metadata = None


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


# Face Detection Setup
# RetinaFace is loaded once and reused across requests for performance.
# If it fails to load, we fall back to Haar cascades automatically.
_retinaface_model = None
_retinaface_available = None  # None = not yet checked, True/False = result


def _load_retinaface():
    """Load the RetinaFace model once. Returns the model or None on failure."""
    global _retinaface_model, _retinaface_available

    if _retinaface_available is not None:
        return _retinaface_model

    try:
        import os as _os
        _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        from retinaface import RetinaFace

        # Build the model by running a dummy detection to trigger weight loading
        RetinaFace.build_model()
        _retinaface_model = RetinaFace
        _retinaface_available = True
        print("RetinaFace loaded successfully (deep learning face detection)")
    except Exception as e:
        print("RetinaFace unavailable, falling back to Haar cascades: {}".format(e))
        _retinaface_model = None
        _retinaface_available = False

    return _retinaface_model


def _detect_faces_retinaface(img):
    """
    Detect faces using RetinaFace deep learning model.

    Returns a list of (x, y, w, h) tuples for each detected face,
    matching the format used by the Haar cascade fallback.
    Returns None if RetinaFace is unavailable (signals fallback to Haar).
    """
    model = _load_retinaface()
    if model is None:
        return None  # Signal to use fallback

    try:
        detections = model.detect_faces(
            img,
            threshold=config.RETINAFACE_CONFIDENCE_THRESHOLD
        )
    except Exception as e:
        # Memory errors or other issues - fall back gracefully
        print("RetinaFace detection failed on image, using Haar fallback: {}".format(e))
        return None

    # RetinaFace returns an empty tuple when no faces found
    if not isinstance(detections, dict) or len(detections) == 0:
        return []

    # Convert from RetinaFace format [x1, y1, x2, y2] to (x, y, w, h)
    faces = []
    for face_data in detections.values():
        area = face_data["facial_area"]
        x1, y1, x2, y2 = int(area[0]), int(area[1]), int(area[2]), int(area[3])
        faces.append((x1, y1, x2 - x1, y2 - y1))

    return faces


def _detect_faces_haar(img):
    """
    Detect faces using Haar cascades (lightweight fallback).

    Checks frontal faces and both left/right profile faces.
    Returns a list of (x, y, w, h) tuples.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frontal_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    all_faces = []

    # Frontal faces
    frontal = frontal_cascade.detectMultiScale(
        grey,
        scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
        minNeighbors=config.FACE_DETECT_MIN_NEIGHBORS,
        minSize=config.FACE_DETECT_MIN_SIZE
    )
    all_faces.extend(frontal)

    # Left-facing profiles
    profiles = profile_cascade.detectMultiScale(
        grey,
        scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
        minNeighbors=config.FACE_DETECT_MIN_NEIGHBORS,
        minSize=config.FACE_DETECT_MIN_SIZE
    )
    all_faces.extend(profiles)

    # Right-facing profiles (detect on flipped image, then mirror coords back)
    flipped = cv2.flip(grey, 1)
    flipped_profiles = profile_cascade.detectMultiScale(
        flipped,
        scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
        minNeighbors=config.FACE_DETECT_MIN_NEIGHBORS,
        minSize=config.FACE_DETECT_MIN_SIZE
    )
    img_width = img.shape[1]
    for (x, y, w, h) in flipped_profiles:
        all_faces.append((img_width - x - w, y, w, h))

    return all_faces


def _blur_faces(img, faces):
    """Apply Gaussian blur to each detected face region in the image."""
    for (x, y, w, h) in faces:
        # Add padding around the face for complete coverage
        pad = int(w * config.FACE_BLUR_PADDING)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        face_region = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(
            face_region, config.FACE_BLUR_KERNEL, config.FACE_BLUR_SIGMA
        )

    return img


def _get_cache_path(image_path):
    """Generate a unique cache path for a blurred image based on its file path."""
    path_hash = hashlib.md5(str(image_path).encode()).hexdigest()
    return config.FACE_CACHE_DIR / "{}.jpg".format(path_hash)


@server.route("/images/<path:p>")
def serve_image(p):
    """Serve images from the data folder, with optional privacy blur."""
    path = IMAGE_FOLDER / p
    if not path.exists():
        return "Not found", 404

    # Check if privacy mode is enabled
    privacy_mode = request.args.get("privacy", "false").lower() == "true"

    if not privacy_mode:
        return send_from_directory(str(path.parent), path.name)

    # Check the cache first to avoid re-detecting and re-blurring
    cache_path = _get_cache_path(path)
    if cache_path.exists():
        return send_from_directory(str(cache_path.parent), cache_path.name)

    # Load image with OpenCV
    img = cv2.imread(str(path))
    if img is None:
        return send_from_directory(str(path.parent), path.name)

    # Detect faces - try RetinaFace first, fall back to Haar cascades
    faces = None
    if config.FACE_DETECT_MODEL == "retinaface":
        faces = _detect_faces_retinaface(img)

    # If RetinaFace was not available or not configured, use Haar cascades
    if faces is None:
        faces = _detect_faces_haar(img)

    # Blur detected faces
    if faces:
        img = _blur_faces(img, faces)

    # Encode to JPEG
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Save to cache so we only blur once per image
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(cache_path), "wb") as f:
            f.write(buffer.tobytes())
    except Exception:
        pass  # Caching is best-effort, don't break serving

    return Response(buffer.tobytes(), mimetype="image/jpeg")


# Component Builders

def create_badge(label, confidence):
    """Create a classification badge using Academic colours."""
    score_pct = confidence * 100
    
    badge_class = "badge-academic"
    
    if confidence >= config.SEARCH_MIN_THRESHOLD:
        badge_class += " badge-high"

    elif confidence >= config.SEARCH_MIN_THRESHOLD - 0.04:
        badge_class += " badge-mid"

    return html.Span(
        "{} ({:.0f}%)".format(label, score_pct),
        className=badge_class + " me-1 mb-1"
    )


# Page Layouts for the Websites

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
                    dbc.Button("Read Documentation", href="https://github.com/730Rashid/Final_Stage_Project_Rashid_LBS-25-986", target="_blank", 
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
                                    html.Div("Embedding Dimensions", className="stat-label")
                                ], className="stat-box")
                            ], width=3),
                            dbc.Col([
                                html.Div([
                                    html.Div(str(len(CLASSIFICATION_LABELS)), className="stat-value"),
                                    html.Div("Classification Labels", className="stat-label")
                                ], className="stat-box")
                            ], width=3),
                        ])
                    ], className="paper-card")
                ], md=12)
            ], className="mb-4"),
            
            # Abstract Text
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H3("Abstract", className="mb-3"),
                        html.P([
                            "When a natural disaster strikes, social media gets swamped with millions of images ",
                            "from the affected areas. For humanitarian organisations, this creates a real problem: ",
                            "how do you find the relevant, useful images amongst all that chaos? ",
                            html.Strong("Traditional keyword searches just don't work"), " because crisis images rarely ",
                            "have descriptive text, and you can't predict in advance what labels you'll need for ",
                            "every possible disaster scenario."
                        ]),
                        html.P([
                            "This project shows how ", html.Strong("zero-shot semantic search"), " using OpenAI's CLIP ",
                            "model can help. By encoding images into a shared embedding space with natural language, ",
                            "we let responders search the dataset using plain English. They can simply describe ",
                            html.Em("what they need to see"), " rather than relying on tags that already exist."
                        ]),
                        html.P([
                            "The visualisation uses ", html.Strong("UMAP dimensionality reduction"), " to project the ",
                            "512-dimensional embedding space into an interactive 2D scatter plot. This reveals the ",
                            "semantic structure of the dataset, showing how images naturally cluster by content type ",
                            "and disaster event."
                        ])
                    ], className="paper-card")
                ], md=8),
                dbc.Col([
                    html.Div([
                        html.H4("How it all works", className="paper-title mb-3"),
                        html.Div([
                            html.Div([
                                html.Strong("1. Data Ingestion"),
                                html.P("CrisisMMD dataset with 17,463 crisis images across 7 events.", 
                                       className="small text-secondary mb-3")
                            ]),
                            html.Div([
                                html.Strong("2. Vectorisation"),
                                html.P("CLIP ViT-B/32 extracts 512-dimensional semantic embeddings.", 
                                       className="small text-secondary mb-3")
                            ]),
                            html.Div([
                                html.Strong("3. Projection"),
                                html.P("UMAP reduces dimensionality for 2D visualisation.", 
                                       className="small text-secondary mb-3")
                            ]),
                            html.Div([
                                html.Strong("4. Search"),
                                html.P("Cosine similarity enables zero-shot text-to-image retrieval.", 
                                       className="small text-secondary mb-0")
                            ])
                        ])
                    ], className="paper-card")
                ], md=4)
            ], className="mb-4"),
            
            # Footer
            html.Footer([
                html.Hr(className="mt-5 mb-4 text-muted opacity-25"),
                html.Div([
                    html.P("Disaster ImageVisualisation Tool", className="fw-bold mb-2", style={"letterSpacing": "0.05em"}),
                    html.P("Built with CLIP, UMAP, and Dash", className="text-secondary small mb-4"),
                    html.Div([
                        html.A([
                            html.I(className="bi bi-github me-2"),
                            "GitHub"
                        ], href="https://github.com/730Rashid", target="_blank",
                           className="text-secondary text-decoration-none me-4"),
                        html.A([
                            html.I(className="bi bi-linkedin me-2"),
                            "LinkedIn"
                        ], href="https://www.linkedin.com/in/rashid-pandor-85537b22b/", 
                           target="_blank", className="text-secondary text-decoration-none")
                    ], className="d-flex justify-content-center align-items-center w-100")
                ], className="d-flex flex-column align-items-center pb-5")
            ])
        ], fluid=True, style={"maxWidth": "1200px"}, className="py-4")
    ])


def explorer_page():
    """Interactive explorer page."""
    # Build cluster filter options if clusters are available
    cluster_options = [{"label": "All Clusters", "value": "all"}]
    if cluster_metadata is not None:
        for cluster_id, info in cluster_metadata.get("clusters", {}).items():
            cluster_id_int = int(cluster_id)
            label = "{} ({})".format(info["name"], info["count"])
            cluster_options.append({"label": label, "value": cluster_id_int})
    
    return dbc.Container([
        # Controls
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
                # Discovered Clusters filter (only if clusters are loaded)
                dbc.Col([
                    dbc.Label("Discovered Cluster", className="small text-secondary fw-bold"),
                    dcc.Dropdown(
                        id="cluster-filter",
                        options=cluster_options,
                        value="all",
                        clearable=False,
                        className="dash-dropdown",
                        disabled=(cluster_metadata is None)
                    )
                ], md=2) if cluster_metadata is not None else dbc.Col([], md=2),
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
                ], md=5),
                dbc.Col([
                    dbc.Label("System Status", className="small text-secondary fw-bold"),
                    html.Div(id="search-status", className="text-secondary small pt-2")
                ], md=2)
            ], align="start", className="g-4")
        ], className="paper-card mb-4 py-3"),
        
        # Privacy Notice (Always On)
        html.Div([
            html.I(className="bi bi-shield-check me-2"),
            html.Span("Privacy", className="fw-bold me-2"),
            html.Span("Faces are automatically blurred to protect victim's privacy.", className="text-muted")
        ], className="alert-privacy mb-4"),
        
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
                ], className="paper-card p-0")
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


def analytics_page():
    """Analytics dashboard for embedding space analysis."""
    return dbc.Container([
        # Page Header
        html.Div([
            html.H2("Embedding Space Analytics", className="mb-2"),
            html.P(
                "Quantitative analysis of the CLIP embedding space across disaster events.",
                className="text-secondary mb-0"
            )
        ], className="paper-card mb-4"),

        # Global Summary Stats
        html.Div(id="analytics-summary", className="mb-4"),

        # Tabs
        dcc.Tabs(
            id="analytics-tabs",
            value="tab-events",
            children=[
                dcc.Tab(label="Event Statistics", value="tab-events",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Embedding Space", value="tab-embedding",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Export", value="tab-export",
                        className="custom-tab", selected_className="custom-tab--selected"),
            ],
            className="mb-4"
        ),

        # Tab Content
        html.Div(id="analytics-tab-content"),

        # Hidden download component
        dcc.Download(id="analytics-download"),

    ], fluid=True, style={"maxWidth": "1200px"}, className="py-4")


# Navigation
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Project Abstract", href="/", active="exact")),
        dbc.NavItem(dbc.NavLink("Data Explorer", href="/explorer", active="exact")),
        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics", active="exact")),
    ],
    brand="Disaster Image Visualisation",
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
    elif pathname == "/analytics":
        return analytics_page()
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


def build_image_card(row, score, score_label="Match", privacy_mode=False):
    """Build an image card with classification badges and caption."""
    try:
        rel_path = Path(row["path"]).relative_to(IMAGE_FOLDER)
        base_url = "/images/{}".format(str(rel_path).replace(os.sep, "/"))

        # Add privacy suffix if enabled
        img_url = "{}?privacy=true".format(base_url) if privacy_mode else base_url

        image_idx = row["original_idx"]
        classifications = classify_image(image_idx, threshold=config.CLASSIFICATION_THRESHOLD)

        badge_elements = [create_badge(label, conf) for label, conf in classifications[:3]]

        score_color = "#16a34a" if score >= 0.30 else "#2563eb"

        # Generate caption using CLIP interrogation
        caption_data = caption_image_by_index(image_idx, style="brief")
        caption_text = caption_data.get("caption", "")

        return html.Div([
            html.Img(src=img_url, className="w-100",
                     style={"borderRadius": "4px 4px 0 0", "objectFit": "cover", "height": "120px"}),
            html.Div([
                html.Small(row["event"], className="text-secondary d-block"),
                html.Span(
                    "{}: {:.0f}%".format(score_label, score * 100),
                    style={"color": score_color, "fontWeight": "600"}
                ),
                html.Div(badge_elements, className="mt-2") if badge_elements else None,
                # Add caption below badges - styled to be subtle and readable
                html.P(
                    caption_text,
                    className="mt-2 mb-0 small",
                    style={"fontStyle": "italic", "color": "#64748b", "lineHeight": "1.4"}
                ) if caption_text and caption_data.get("available") else None
            ], className="p-2", style={"borderTop": "1px solid #e2e8f0"})
        ], className="paper-card p-0 mb-3")
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
        Input("cluster-filter", "value"),
        Input("clicked-point-store", "data")
    ],
    [State("search-input", "value")]
)
def update_view(n_clicks, n_submit, selected_event, selected_cluster, clicked_index, query):
    """Update the visualisation based on user interaction."""
    # Privacy is always enabled
    privacy_mode = True
    fig = go.Figure()
    images = []
    status = "Ready"
    gallery_title = "Results"
    
    # Start with full dataframe
    working_df = df.copy()
    filter_parts = []
    
    # Apply event filter
    if selected_event and selected_event != "all":
        working_df = working_df[working_df["event"] == selected_event]
        filter_parts.append(selected_event)
    
    # Apply cluster filter
    if selected_cluster is not None and selected_cluster != "all" and "cluster_id" in df.columns:
        working_df = working_df[working_df["cluster_id"] == selected_cluster]
        if cluster_metadata:
            cluster_info = cluster_metadata.get("clusters", {}).get(str(selected_cluster), {})
            cluster_name = cluster_info.get("name", "Cluster {}".format(selected_cluster))
            filter_parts.append(cluster_name)
    
    # Create ghosted dataframe (points not in current filter)
    filtered_indices_set = set(working_df.index)
    ghosted_df = df[~df.index.isin(filtered_indices_set)]
    filtered_df = working_df
    filtered_indices = filtered_df["original_idx"].tolist() if len(filtered_df) < len(df) else None
    
    # Build status message
    if filter_parts:
        status = "{:,} images in {}".format(len(filtered_df), " ∩ ".join(filter_parts))
    else:
        status = "{:,} images".format(len(df))
    
    # Ghost layer
    if len(ghosted_df) > 0:
        fig.add_trace(go.Scattergl(
            x=ghosted_df["x"],
            y=ghosted_df["y"],
            mode="markers",
            marker=dict(size=3, color="#d1d5db", opacity=0.3),
            hoverinfo="skip",
            showlegend=False
        ))
    
    # Active layer
    fig.add_trace(go.Scattergl(
        x=filtered_df["x"],
        y=filtered_df["y"],
        mode="markers",
        marker=dict(size=4, color="#94a3b8", opacity=0.5),
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
                size=14, color="#ea580c", opacity=1.0,
                line=dict(width=2, color="white"), symbol="star"
            ),
            hovertemplate="<b>Query Image</b><extra></extra>",
            showlegend=False
        ))
        
        fig.add_trace(go.Scattergl(
            x=match_df["x"],
            y=match_df["y"],
            mode="markers",
            marker=dict(
                size=6, color="#2563eb", opacity=0.8,
                line=dict(width=0)
            ),
            text=match_df["hover"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        ))
        
        status = "Identified {} visually similar images.".format(len(indices))
        gallery_title = "Visually Similar Images"
        
        for _, row in match_df.head(10).iterrows():
            card = build_image_card(row, row["score"], "Similarity", privacy_mode)
            if card:
                images.append(card)

    # Text Search
    elif query and len(query.strip()) >= config.SEARCH_MIN_QUERY_LENGTH:
        trimmed = query.strip()[:config.SEARCH_MAX_QUERY_LENGTH]
        indices, scores = semantic_search(trimmed, subset_indices=filtered_indices)
        match_df = df.iloc[indices].copy()
        match_df["score"] = scores

        strong_matches = match_df[match_df["score"] >= config.SEARCH_MIN_THRESHOLD]
        
        if len(strong_matches) > 0:
            fig.add_trace(go.Scattergl(
                x=strong_matches["x"],
                y=strong_matches["y"],
                mode="markers",
                marker=dict(
                    size=8, color="#2563eb", opacity=0.9,
                    line=dict(width=1, color="white")
                ),
                text=strong_matches["hover"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False
            ))
            
            status = "Found {} matches for '{}'.".format(len(strong_matches), trimmed)
            gallery_title = "Search Results: {}".format(trimmed)
            
            for _, row in strong_matches.head(10).iterrows():
                card = build_image_card(row, row["score"], "Match", privacy_mode)
                if card:
                    images.append(card)
        else:
            status = "No significant matches found for '{}'.".format(trimmed)
            gallery_title = "No Results"
            images.append(html.Div([
                html.P("No images matched your query with sufficient confidence.", 
                       className="text-secondary"),
                html.Small("Try using broader terms like 'flooding' or 'rubble'.", className="text-muted")
            ], className="text-center py-5"))

    fig.update_layout(
        plot_bgcolor="#ffffff",
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
    
    # Show sample images if no search/click is active
    if not images and not query and clicked_index is None:
        gallery_title = "Sample Images"
        status = "Click a point or search to explore. Showing samples from each event."
        
        # Get one sample from each event category
        for event in UNIQUE_EVENTS:
            event_images = filtered_df[filtered_df["event"] == event]
            
            if len(event_images) > 0:
                sample_row = event_images.sample(1).iloc[0]
                card = build_image_card(sample_row, 1.0, "Sample", privacy_mode)
                
                if card:
                    images.append(card)
        
        final_grid = [dbc.Col(img, width=12) for img in images]
        image_grid = dbc.Row(final_grid, className="g-3")

    return fig, image_grid, status, gallery_title


# Analytics Callbacks

@app.callback(
    Output("analytics-summary", "children"),
    Input("url", "pathname")
)
def render_analytics_summary(pathname):
    """Render global summary stat cards on the analytics page."""
    if pathname != "/analytics":
        return []

    analytics = get_analytics()
    summary = analytics.global_summary()

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("{:,}".format(summary["total_images"]), className="stat-value"),
                    html.Div("Total Images", className="stat-label")
                ], className="stat-box")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div(str(summary["embedding_dim"]), className="stat-value"),
                    html.Div("Embedding Dimensions", className="stat-label")
                ], className="stat-box")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div("{:.3f}".format(summary["global_mean_similarity"]), className="stat-value"),
                    html.Div("Mean Pairwise Similarity", className="stat-label")
                ], className="stat-box")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div("{:.3f}".format(summary["global_std_similarity"]), className="stat-value"),
                    html.Div("Similarity Std Dev", className="stat-label")
                ], className="stat-box")
            ], width=3),
        ])
    ], className="paper-card")


@app.callback(
    Output("analytics-tab-content", "children"),
    Input("analytics-tabs", "value")
)
def render_analytics_tab(tab):
    """Render the selected analytics tab content."""
    analytics = get_analytics()

    if tab == "tab-events":
        return _build_events_tab(analytics)
    elif tab == "tab-embedding":
        return _build_embedding_tab(analytics)
    elif tab == "tab-export":
        return _build_export_tab()
    return []


def _build_events_tab(analytics):
    """Build the Event Statistics tab content."""
    stats = analytics.per_event_stats()
    events = list(stats.keys())
    counts = [stats[e]["count"] for e in events]
    cohesions = [stats[e]["cohesion"] for e in events]
    spreads = [stats[e]["spread"] for e in events]

    # Image count bar chart
    count_fig = go.Figure(data=[
        go.Bar(
            x=events, y=counts,
            marker_color="#2563eb",
            text=counts, textposition="outside"
        )
    ])
    count_fig.update_layout(
        title="Image Count per Event",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=40, r=20, t=50, b=80),
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="Count", gridcolor="#f1f5f9")
    )

    # Cohesion bar chart
    cohesion_fig = go.Figure(data=[
        go.Bar(
            x=events, y=cohesions,
            marker_color="#16a34a",
            text=["{:.4f}".format(c) for c in cohesions],
            textposition="outside",
            error_y=dict(type="data", array=spreads, visible=True, color="#94a3b8")
        )
    ])
    cohesion_fig.update_layout(
        title="Intra-Event Cohesion (Mean Pairwise Cosine Similarity)",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=40, r=20, t=50, b=80),
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="Cosine Similarity", gridcolor="#f1f5f9")
    )

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=count_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=cohesion_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=6),
        ], className="g-4")
    ])


def _build_embedding_tab(analytics):
    """Build the Embedding Space tab content."""
    # Inter-event similarity heatmap
    matrix = analytics.inter_event_similarity_matrix()
    events = analytics.events

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=events, y=events,
        colorscale="Blues",
        zmin=0.5, zmax=1.0,
        text=[["{:.3f}".format(v) for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Similarity: %{z:.4f}<extra></extra>"
    ))
    heatmap_fig.update_layout(
        title="Inter-Event Centroid Cosine Similarity",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=120, r=20, t=50, b=120),
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange="reversed"),
        width=600, height=500
    )

    # Intra-event box plots
    distributions = analytics.intra_event_distributions()
    box_fig = go.Figure()
    colours = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#dc2626", "#0891b2", "#ca8a04"]
    for i, event in enumerate(events):
        box_fig.add_trace(go.Box(
            y=distributions[event],
            name=event,
            marker_color=colours[i % len(colours)],
            boxmean=True
        ))
    box_fig.update_layout(
        title="Intra-Event Pairwise Similarity Distributions",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=40, r=20, t=50, b=80),
        yaxis=dict(title="Cosine Similarity", gridcolor="#f1f5f9"),
        showlegend=False
    )

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=heatmap_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=box_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=6),
        ], className="g-4")
    ])


def _build_export_tab():
    """Build the Export tab content."""
    return html.Div([
        html.Div([
            html.H4("Export Analytics Report", className="paper-title mb-3"),
            html.P(
                "Download all embedding space analytics as a JSON report. "
                "Includes per-event statistics, inter-event similarity matrix, "
                "and global summary metrics.",
                className="text-secondary"
            ),
            dbc.Button(
                [html.I(className="bi bi-download me-2"), "Download JSON Report"],
                id="btn-export-json",
                className="btn-primary-action",
                n_clicks=0
            ),
        ], className="paper-card", style={"maxWidth": "600px"})
    ])


@app.callback(
    Output("analytics-download", "data"),
    Input("btn-export-json", "n_clicks"),
    prevent_initial_call=True
)
def download_analytics_report(n_clicks):
    """Generate and send the analytics JSON report."""
    analytics = get_analytics()
    report = analytics.export_report()
    return dcc.send_string(
        json.dumps(report, indent=2),
        filename="embedding_analytics_report.json"
    )


if __name__ == "__main__":
    print("\nServer running at http://127.0.0.1:8050/")
    app.run(debug=False)
