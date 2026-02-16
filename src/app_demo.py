"""
CrisisMMD Visualisation App - Frontend.

A humanitarian aid tool for exploring disaster imagery using AI-powered
semantic search and visual similarity. Built with CLIP embeddings and UMAP.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

# To run the script do this: .venv/Scripts/python.exe src/app_demo.py


import sys
import json
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
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
    get_topology_analytics,
    semantic_search,
    visual_search,
    classify_image,
    caption_image_by_index,
    get_damage_severity,
    get_heatmap_bytes,
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
# YuNet is OpenCV's built-in deep learning face detector (~230KB model).
# It's loaded once and reused across all requests for performance.
# If YuNet fails to load we then fall back to Haar cascades automatically.
_yunet_detector = None
_yunet_available = None  # None = not yet checked, True/False = result


def _load_yunet():
    """
    Load the YuNet face detector once.

    YuNet is a lightweight CNN-based face detector bundled with OpenCV.
    It's much more accurate than Haar cascades while using minimal memory.
    Returns the detector instance or None on failure.
    """
    global _yunet_detector, _yunet_available

    if _yunet_available is not None:
        return _yunet_detector

    model_path = str(config.YUNET_MODEL_PATH)

    if not config.YUNET_MODEL_PATH.exists():
        print("YuNet model not found at {}, falling back to Haar cascades".format(model_path))
        _yunet_available = False
        return None

    try:
        # Create detector with a placeholder input size (updated per image later)
        _yunet_detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            config.YUNET_CONFIDENCE_THRESHOLD,
            config.YUNET_NMS_THRESHOLD,
            5000
        )
        _yunet_available = True
        print("YuNet face detector loaded (deep learning, ~230KB model)")
    except Exception as e:
        print("YuNet unavailable, falling back to Haar cascades: {}".format(e))
        _yunet_detector = None
        _yunet_available = False

    return _yunet_detector


def _detect_faces_yunet(img):
    """
    Detect faces using YuNet deep learning model (built into OpenCV).

    Returns a list of (x, y, w, h) tuples for each detected face,
    matching the format used by the Haar cascade fallback.
    Returns None if YuNet is unavailable (signals fallback to Haar).
    """
    detector = _load_yunet()
    if detector is None:
        return None

    try:
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, detections = detector.detect(img)
    except Exception as e:
        print("YuNet detection failed, using Haar fallback: {}".format(e))
        return None

    if detections is None or len(detections) == 0:
        return []

    # YuNet returns [x, y, w, h, ..., score] per face (14 or 15 values)
    faces = []
    for face in detections:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        faces.append((x, y, fw, fh))

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
    try:
        frontal = frontal_cascade.detectMultiScale(
            grey,
            scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
            minNeighbors=config.FACE_DETECT_MIN_NEIGHBORS,
            minSize=config.FACE_DETECT_MIN_SIZE
        )
        all_faces.extend(frontal)
    except Exception:
        pass

    # Left-facing profiles
    try:
        profiles = profile_cascade.detectMultiScale(
            grey,
            scaleFactor=config.FACE_DETECT_SCALE_FACTOR,
            minNeighbors=config.FACE_DETECT_MIN_NEIGHBORS,
            minSize=config.FACE_DETECT_MIN_SIZE
        )
        all_faces.extend(profiles)
    except Exception:
        pass

    # Right-facing profiles (detect on flipped image, then mirror coords back)
    try:
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
    except Exception:
        pass

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

    # Detect faces — try YuNet first, fall back to Haar cascades
    try:
        faces = None
        if config.FACE_DETECT_MODEL == "yunet":
            faces = _detect_faces_yunet(img)

        # If YuNet was not available or not configured then use Haar cascades
        if faces is None:
            faces = _detect_faces_haar(img)

        # Blur detected faces
        if faces:
            img = _blur_faces(img, faces)
            
    except Exception:
        pass  # Face detection failure — serve image unblurred rather than 500

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


@server.route("/heatmap/<path:p>")
def serve_heatmap(p):
    """Serve CLIP attention heatmap for an image."""
    path = IMAGE_FOLDER / p
    if not path.exists():
        return "Not found", 404
    
    try:
        img_bytes = get_heatmap_bytes(str(path))
        return Response(img_bytes, mimetype="image/jpeg")
    
    except Exception as e:
        return "Heatmap error: {}".format(e), 500


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
                dcc.Tab(label="Cross-Disaster Transfer", value="tab-transfer",
                        className="custom-tab", selected_className="custom-tab--selected"),
                dcc.Tab(label="Topology", value="tab-topology",
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
        heatmap_url = "/heatmap/{}".format(str(rel_path).replace(os.sep, "/"))

        # Add privacy suffix if enabled
        img_url = "{}?privacy=true".format(base_url) if privacy_mode else base_url

        image_idx = row["original_idx"]
        classifications = classify_image(image_idx, threshold=config.CLASSIFICATION_THRESHOLD)

        badge_elements = [create_badge(label, conf) for label, conf in classifications[:3]]

        score_color = "#16a34a" if score >= 0.30 else "#2563eb"

        # Damage severity score
        severity = get_damage_severity(image_idx)
        severity_badge = html.Span(
            "{} {:.0f}%".format(severity["category"], severity["score"] * 100),
            className="badge me-1",
            style={"backgroundColor": severity["color"], "fontSize": "0.7rem", "color": "#fff"}
        )

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
                html.Div(severity_badge, className="mt-1"),
                html.Div(badge_elements, className="mt-1") if badge_elements else None,
                # Add caption below badges - styled to be subtle and readable
                html.P(
                    caption_text,
                    className="mt-2 mb-0 small",
                    style={"fontStyle": "italic", "color": "#64748b", "lineHeight": "1.4"}
                ) if caption_text and caption_data.get("available") else None,
                html.A(
                    "View Attention Heatmap",
                    href=heatmap_url,
                    target="_blank",
                    className="small d-block mt-2",
                    style={"color": "#64748b", "textDecoration": "none", "opacity": "0.75"}
                )
            ], className="p-2", style={"borderTop": "1px solid #e2e8f0"})
        ], className="paper-card p-0 mb-3")
    except Exception:
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
    
    elif tab == "tab-transfer":
        return _build_transfer_tab(analytics)

    elif tab == "tab-topology":
        return _build_topology_tab()

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


def _build_transfer_tab(analytics):
    """Build the Cross-Disaster Transfer tab content."""
    events = analytics.events
    type_groups = config.DISASTER_TYPE_GROUPS

    # Colour map for disaster types
    type_colours = {
        "Wildfire": "#ea580c",
        "Hurricane/Flood": "#2563eb",
        "Earthquake": "#7c3aed",
    }

    # Directional retrieval heatmap
    retrieval = analytics.cross_event_retrieval_matrix()
    retrieval_fig = go.Figure(data=go.Heatmap(
        z=retrieval,
        x=events, y=events,
        colorscale="RdBu",
        zmid=float(np.mean(retrieval)),
        text=[["{:.3f}".format(v) for v in row] for row in retrieval],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate=(
            "<b>Source:</b> %{y}<br>"
            "<b>Target:</b> %{x}<br>"
            "Mean Similarity: %{z:.4f}<extra></extra>"
        )
    ))
    retrieval_fig.update_layout(
        title="Cross-Event Retrieval Transfer (Source → Target Centroid Similarity)",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=140, r=20, t=50, b=120),
        xaxis=dict(title="Target Event", tickangle=-30),
        yaxis=dict(title="Source Event", autorange="reversed"),
        height=520
    )

    # LOO classification bar chart
    loo = analytics.loo_classification_accuracy()
    event_accs = loo["event_accuracies"]
    overall_acc = loo["overall_accuracy"]

    loo_events = list(event_accs.keys())
    loo_values = [event_accs[e] * 100 for e in loo_events]
    loo_colours = [type_colours.get(type_groups.get(e, ""), "#94a3b8") for e in loo_events]

    loo_fig = go.Figure(data=[
        go.Bar(
            x=loo_events, y=loo_values,
            marker_color=loo_colours,
            text=["{:.1f}%".format(v) for v in loo_values],
            textposition="outside"
        )
    ])
    loo_fig.add_hline(
        y=overall_acc * 100, line_dash="dash", line_color="#64748b",
        annotation_text="Overall: {:.1f}%".format(overall_acc * 100),
        annotation_position="top right"
    )
    loo_fig.update_layout(
        title="Leave-One-Out Classification by Disaster Type",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=40, r=20, t=50, b=80),
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="Accuracy (%)", gridcolor="#f1f5f9", range=[0, 110])
    )

    # Disaster-type grouping chart
    grouping = analytics.disaster_type_grouping_analysis()
    within = grouping["within_type"]
    across = grouping["overall_across"]
    separation = grouping["separation_ratio"]

    type_names = sorted(within.keys())
    within_vals = []
    bar_texts = []
    bar_colours_group = []
    
    for t in type_names:
        v = within[t]
        
        if v is not None:
            within_vals.append(v)
            bar_texts.append("{:.4f}".format(v))
            
        else:
            within_vals.append(0)
            bar_texts.append("N/A (1 event)")
            
        bar_colours_group.append(type_colours.get(t, "#94a3b8"))

    group_fig = go.Figure(data=[
        go.Bar(
            x=type_names, y=within_vals,
            marker_color=bar_colours_group,
            text=bar_texts, textposition="outside"
        )
    ])
    if across is not None:
        group_fig.add_hline(
            y=across, line_dash="dot", line_color="#dc2626",
            annotation_text="Across-type avg: {:.4f}".format(across),
            annotation_position="top right"
        )
    sep_text = "{:.2f}x".format(separation) if separation is not None else "N/A"
    group_fig.update_layout(
        title="Within-Type vs Across-Type Similarity (Separation: {})".format(sep_text),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=40, r=20, t=50, b=60),
        yaxis=dict(title="Mean Cosine Similarity", gridcolor="#f1f5f9")
    )

    return html.Div([
        # Explanation card
        html.Div([
            html.H4("Cross-Disaster Transfer Analysis", className="paper-title mb-3"),
            html.P([
                "This tab analyses how CLIP representations transfer across disaster events and types. ",
                "The ",
                html.Strong("retrieval heatmap"),
                " shows directional similarity (source images → target centroid). ",
                "The ",
                html.Strong("LOO classification"),
                " tests if held-out event images can be classified by disaster type using remaining centroids. ",
                "The ",
                html.Strong("type grouping"),
                " compares within-type vs across-type similarity to measure semantic clustering."
            ], className="text-secondary mb-0")
        ], className="paper-card mb-4"),

        # Full-width retrieval heatmap
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=retrieval_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=12),
        ], className="g-4 mb-4"),

        # Bottom row: LOO + Grouping
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=loo_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=6),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=group_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=6),
        ], className="g-4")
    ])


def _build_topology_tab():
    """Build the Topology tab: Persistent Homology + Ollivier-Ricci Curvature."""
    topo = get_topology_analytics()

    # ------------------------------------------------------------------
    # Phase 1: Persistence Diagram (H0 + H1)
    # ------------------------------------------------------------------
    ph   = topo.persistence_homology()
    h0   = ph["h0"]
    h1   = ph["h1"]

    # Build persistence diagram scatter plot
    # Each point is (birth, death); distance from the diagonal = persistence
    max_val = 0.0
    if h0["death"]:
        max_val = max(max_val, max(h0["death"]))
    if h1["death"]:
        max_val = max(max_val, max(h1["death"]))
    max_val = max_val * 1.05 or 1.0   # avoid zero range

    pd_fig = go.Figure()

    # Diagonal: birth == death (zero-persistence, i.e. noise)
    pd_fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#94a3b8", dash="dash", width=1),
        name="Diagonal (noise)",
        hoverinfo="skip",
    ))

    # H0 features — connected components
    if h0["birth"]:
        h0_sizes = [6 + 20 * p / (h0["max_persistence"] + 1e-9)
                    for p in h0["persistence"]]
        pd_fig.add_trace(go.Scatter(
            x=h0["birth"], y=h0["death"],
            mode="markers",
            marker=dict(
                color="#2563eb", size=h0_sizes, opacity=0.75,
                line=dict(width=1, color="#1e40af"),
            ),
            name="H0 — Connected components ({} features)".format(h0["n_features"]),
            hovertemplate=(
                "H0 feature<br>"
                "Birth: %{x:.4f}<br>"
                "Death: %{y:.4f}<br>"
                "Persistence: %{customdata:.4f}<extra></extra>"
            ),
            customdata=h0["persistence"],
        ))

    # H1 features — loops / cycles
    if h1["birth"]:
        h1_sizes = [8 + 24 * p / (h1["max_persistence"] + 1e-9)
                    for p in h1["persistence"]]
        pd_fig.add_trace(go.Scatter(
            x=h1["birth"], y=h1["death"],
            mode="markers",
            marker=dict(
                color="#dc2626", size=h1_sizes, opacity=0.75,
                symbol="diamond",
                line=dict(width=1, color="#991b1b"),
            ),
            name="H1 — Loops / cycles ({} features)".format(h1["n_features"]),
            hovertemplate=(
                "H1 feature<br>"
                "Birth: %{x:.4f}<br>"
                "Death: %{y:.4f}<br>"
                "Persistence: %{customdata:.4f}<extra></extra>"
            ),
            customdata=h1["persistence"],
        ))

    pd_fig.update_layout(
        title="Persistence Diagram — Vietoris-Rips on {} images (cosine distance)".format(
            ph["sample_size"]),
        xaxis=dict(title="Birth radius", gridcolor="#f1f5f9", range=[0, max_val]),
        yaxis=dict(title="Death radius", gridcolor="#f1f5f9", range=[0, max_val]),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        height=480,
    )

    # ------------------------------------------------------------------
    # Phase 2: Ollivier-Ricci Curvature — Histogram
    # ------------------------------------------------------------------
    ricci = topo.ollivier_ricci_curvature()
    curvs = ricci["edge_curvatures"]

    # Build a colour-coded histogram: red bins (κ < 0) vs blue bins (κ > 0)
    curv_arr = np.array(curvs)
    neg_vals = curv_arr[curv_arr < 0].tolist()
    pos_vals = curv_arr[curv_arr >= 0].tolist()

    hist_fig = go.Figure()
    if neg_vals:
        hist_fig.add_trace(go.Histogram(
            x=neg_vals,
            name="Negative κ — bridge / bottleneck ({:.1f}%)".format(
                ricci["pct_negative"]),
            marker_color="#dc2626", opacity=0.8,
            nbinsx=30,
            hovertemplate="κ range: %{x}<br>Edges: %{y}<extra></extra>",
        ))
    if pos_vals:
        hist_fig.add_trace(go.Histogram(
            x=pos_vals,
            name="Positive κ — dense cluster ({:.1f}%)".format(
                ricci["pct_positive"]),
            marker_color="#2563eb", opacity=0.8,
            nbinsx=30,
            hovertemplate="κ range: %{x}<br>Edges: %{y}<extra></extra>",
        ))

    hist_fig.add_vline(
        x=0, line_dash="dash", line_color="#64748b",
        annotation_text="κ = 0 (flat)",
        annotation_position="top right",
    )
    hist_fig.add_vline(
        x=ricci["global_mean"], line_dash="dot", line_color="#7c3aed",
        annotation_text="Mean κ = {:.3f}".format(ricci["global_mean"]),
        annotation_position="top left",
    )
    hist_fig.update_layout(
        title="Ollivier-Ricci Curvature Histogram — {}-NN graph on {} images".format(
            ricci["k_neighbors"], ricci["sample_size"]),
        barmode="overlay",
        xaxis=dict(title="Curvature κ(u, v)", gridcolor="#f1f5f9"),
        yaxis=dict(title="Number of edges", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        height=380,
    )

    # ------------------------------------------------------------------
    # Phase 2b: Per-event mean curvature bar chart
    # ------------------------------------------------------------------
    type_groups = config.DISASTER_TYPE_GROUPS
    type_colours = {
        "Wildfire":       "#ea580c",
        "Hurricane/Flood": "#2563eb",
        "Earthquake":     "#7c3aed",
    }

    event_curv = ricci["event_mean_curvature"]
    ev_names   = sorted(event_curv.keys())
    ev_vals    = [event_curv[e] for e in ev_names]
    ev_colours = [
        type_colours.get(type_groups.get(e, ""), "#94a3b8")
        for e in ev_names
    ]

    ev_fig = go.Figure(data=[go.Bar(
        x=ev_names, y=ev_vals,
        marker_color=ev_colours,
        text=["{:.3f}".format(v) for v in ev_vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Mean κ: %{y:.4f}<extra></extra>",
    )])
    ev_fig.add_hline(
        y=0, line_color="#64748b", line_dash="dash",
        annotation_text="κ = 0",
        annotation_position="top right",
    )
    ev_fig.update_layout(
        title="Mean Intra-Event Curvature per Disaster",
        xaxis=dict(tickangle=-30),
        yaxis=dict(title="Mean κ", gridcolor="#f1f5f9"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans"),
        margin=dict(l=50, r=20, t=50, b=100),
        height=380,
    )

    # ------------------------------------------------------------------
    # Summary metric cards
    # ------------------------------------------------------------------
    def _metric_card(label, value, sub=""):
        return html.Div([
            html.P(label, className="text-secondary mb-1",
                   style={"fontSize": "0.75rem", "textTransform": "uppercase",
                          "letterSpacing": "0.05em"}),
            html.H4(value, className="mb-0",
                    style={"fontWeight": "700", "fontFamily": "IBM Plex Mono"}),
            html.P(sub, className="text-secondary mb-0",
                   style={"fontSize": "0.8rem"}),
        ], className="paper-card text-center", style={"padding": "1rem"})

    geo_type = (
        "Hyperbolic / tree-like" if ricci["global_mean"] < -0.05
        else "Spherical / cluster-like" if ricci["global_mean"] > 0.05
        else "Flat (Euclidean-like)"
    )

    return html.Div([
        # Explanation card
        html.Div([
            html.H4("Topological Analysis of the CLIP Embedding Space",
                    className="paper-title mb-3"),
            html.P([
                "This tab applies two techniques from algebraic topology and discrete "
                "geometry to characterise the global structure of the disaster embedding space. ",
                html.Strong("Persistent homology"),
                " (Vietoris-Rips filtration) reveals how many distinct semantic clusters "
                "exist and whether any topological 'holes' are present — regions of semantic "
                "ambiguity not captured by any single cluster. ",
                html.Strong("Ollivier-Ricci curvature"),
                " measures the local geometry of the embedding graph: positive curvature "
                "indicates dense, sphere-like clusters, while negative curvature identifies "
                "bridge edges connecting semantically distant regions.",
            ], className="text-secondary mb-0"),
        ], className="paper-card mb-4"),

        # Summary metric cards
        dbc.Row([
            dbc.Col(_metric_card(
                "H0 Persistent Features",
                str(h0["n_features"]),
                "Semantic components (finite lifetime)"
            ), md=3),
            dbc.Col(_metric_card(
                "H1 Persistent Features",
                str(h1["n_features"]),
                "Topological loops in embedding space"
            ), md=3),
            dbc.Col(_metric_card(
                "Global Mean Curvature",
                "{:.3f}".format(ricci["global_mean"]),
                geo_type
            ), md=3),
            dbc.Col(_metric_card(
                "Bridge Edges (κ < 0)",
                "{:.1f}%".format(ricci["pct_negative"]),
                "of {} total edges".format(ricci["n_edges"])
            ), md=3),
        ], className="g-3 mb-4"),

        # Persistence diagram (full width)
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=pd_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=12),
        ], className="g-4 mb-4"),

        # Curvature histogram + per-event bar chart
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=hist_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=7),
            dbc.Col([
                html.Div([
                    dcc.Graph(figure=ev_fig, config={"displaylogo": False})
                ], className="paper-card")
            ], md=5),
        ], className="g-4"),
    ])


def _build_export_tab():
    """Build the Export tab with a live preview of key findings."""
    analytics = get_analytics()

    # Gather all data for the preview
    summary = analytics.global_summary()
    stats = analytics.per_event_stats()
    loo = analytics.loo_classification_accuracy()
    grouping = analytics.disaster_type_grouping_analysis()

    events = sorted(stats.keys())
    type_groups = config.DISASTER_TYPE_GROUPS

    # --- Key Findings Preview ---
    # Global overview cards
    global_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div("Total Images", className="text-secondary",
                      style={"fontSize": "0.8rem", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div("{:,}".format(summary["total_images"]),
                      style={"fontSize": "1.8rem", "fontWeight": "600", "color": "#1e293b"}),
        ], className="paper-card text-center py-3"), md=3),
        dbc.Col(html.Div([
            html.Div("Embedding Dim", className="text-secondary",
                      style={"fontSize": "0.8rem", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div(str(summary["embedding_dim"]),
                      style={"fontSize": "1.8rem", "fontWeight": "600", "color": "#1e293b"}),
        ], className="paper-card text-center py-3"), md=3),
        dbc.Col(html.Div([
            html.Div("Global Mean Sim", className="text-secondary",
                      style={"fontSize": "0.8rem", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div("{:.4f}".format(summary["global_mean_similarity"]),
                      style={"fontSize": "1.8rem", "fontWeight": "600", "color": "#2563eb"}),
        ], className="paper-card text-center py-3"), md=3),
        dbc.Col(html.Div([
            html.Div("LOO Accuracy", className="text-secondary",
                      style={"fontSize": "0.8rem", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div("{:.1f}%".format(loo["overall_accuracy"] * 100),
                      style={"fontSize": "1.8rem", "fontWeight": "600", "color": "#16a34a"}),
        ], className="paper-card text-center py-3"), md=3),
    ], className="g-3 mb-4")

    # Per-event summary table
    table_rows = []
    for event in events:
        s = stats[event]
        acc = loo["event_accuracies"].get(event, 0)
        dtype = type_groups.get(event, "Unknown")
        table_rows.append(html.Tr([
            html.Td(event, style={"fontWeight": "500"}),
            html.Td(dtype),
            html.Td("{:,}".format(s["count"])),
            html.Td("{:.4f}".format(s["cohesion"])),
            html.Td("{:.4f}".format(s["spread"])),
            html.Td("{:.1f}%".format(acc * 100)),
        ]))

    event_table = html.Div([
        html.H5("Per-Event Summary", className="paper-title mb-3"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Event"), html.Th("Type"), html.Th("Images"),
                html.Th("Cohesion"), html.Th("Spread"), html.Th("LOO Acc"),
            ])),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, size="sm",
           style={"fontSize": "0.85rem"})
    ], className="paper-card mb-4")

    # Type grouping summary
    within = grouping["within_type"]
    sep = grouping["separation_ratio"]
    grouping_items = []
    for t in sorted(within.keys()):
        v = within[t]
        val_str = "{:.4f}".format(v) if v is not None else "N/A (single event)"
        grouping_items.append(html.Li([
            html.Strong(t), ": within-type similarity = {}".format(val_str)
        ]))

    type_summary = html.Div([
        html.H5("Disaster Type Grouping", className="paper-title mb-3"),
        html.Ul(grouping_items, style={"fontSize": "0.9rem", "lineHeight": "1.8"}),
        html.P([
            "Across-type similarity: ",
            html.Strong("{:.4f}".format(grouping["overall_across"])) if grouping["overall_across"] else "N/A",
            " | Separation ratio: ",
            html.Strong("{:.2f}x".format(sep)) if sep else "N/A",
        ], className="text-secondary", style={"fontSize": "0.9rem"})
    ], className="paper-card mb-4")

    # Download section
    download_section = html.Div([
        html.H5("Download Report", className="paper-title mb-3"),
        html.P(
            "Export the full analytics report with all metrics, matrices, "
            "and cross-disaster transfer findings.",
            className="text-secondary mb-3"
        ),
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-filetype-json me-2"), "Download JSON"],
                    id="btn-export-json",
                    className="btn-primary-action w-100",
                ),
                html.Small("Structured data for further analysis",
                           className="text-secondary d-block mt-1 text-center")
            ], md=4),
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-file-text me-2"), "Download Text Summary"],
                    id="btn-export-txt",
                    className="btn-primary-action w-100",
                    style={"backgroundColor": "#475569"}
                ),
                html.Small("Human-readable findings for your dissertation",
                           className="text-secondary d-block mt-1 text-center")
            ], md=4),
        ], className="g-3")
    ], className="paper-card")

    return html.Div([global_cards, event_table, type_summary, download_section])


def _generate_text_report(analytics) -> str:
    """Generate a human-readable plain-text analytics report."""
    summary = analytics.global_summary()
    stats = analytics.per_event_stats()
    loo = analytics.loo_classification_accuracy()
    grouping = analytics.disaster_type_grouping_analysis()
    type_groups = config.DISASTER_TYPE_GROUPS
    events = analytics.events

    lines = []
    lines.append("=" * 68)
    lines.append("  CLIP EMBEDDING SPACE ANALYTICS REPORT")
    lines.append("  Visualising Natural Disaster Image Embeddings")
    lines.append("  Model: CLIP ViT-B/32  |  Dataset: CrisisMMD")
    lines.append("=" * 68)
    lines.append("")

    # Section 1: Global Summary
    lines.append("1. GLOBAL EMBEDDING SPACE SUMMARY")
    lines.append("-" * 40)
    lines.append("  Total images:          {:,}".format(summary["total_images"]))
    lines.append("  Embedding dimensions:  {}".format(summary["embedding_dim"]))
    lines.append("  Number of events:      {}".format(summary["num_events"]))
    lines.append("  Global mean similarity: {:.4f}".format(summary["global_mean_similarity"]))
    lines.append("  Global std similarity:  {:.4f}".format(summary["global_std_similarity"]))
    lines.append("  Similarity range:       [{:.4f}, {:.4f}]".format(
        summary["global_min_similarity"], summary["global_max_similarity"]))
    lines.append("")

    # Section 2: Per-Event Statistics
    lines.append("2. PER-EVENT STATISTICS")
    lines.append("-" * 40)
    header = "  {:<25s} {:>6s} {:>10s} {:>10s} {:>8s}".format(
        "Event", "Count", "Cohesion", "Spread", "LOO Acc")
    lines.append(header)
    lines.append("  " + "-" * 61)
    for event in events:
        s = stats[event]
        acc = loo["event_accuracies"].get(event, 0)
        lines.append("  {:<25s} {:>6,d} {:>10.4f} {:>10.4f} {:>7.1f}%".format(
            event, s["count"], s["cohesion"], s["spread"], acc * 100))
    lines.append("")

    # Section 3: Cross-Disaster Transfer
    lines.append("3. CROSS-DISASTER TRANSFER ANALYSIS")
    lines.append("-" * 40)
    lines.append("")
    lines.append("  Leave-One-Out Classification (by disaster type):")
    lines.append("  Overall accuracy: {:.1f}%".format(loo["overall_accuracy"] * 100))
    lines.append("")
    for dtype, acc in sorted(loo["type_accuracies"].items()):
        lines.append("    {:<20s} {:.1f}%".format(dtype, acc * 100))
    lines.append("")

    # Section 4: Disaster Type Grouping
    lines.append("4. DISASTER TYPE GROUPING")
    lines.append("-" * 40)
    within = grouping["within_type"]
    for dtype in sorted(within.keys()):
        v = within[dtype]
        val = "{:.4f}".format(v) if v is not None else "N/A (single event)"
        lines.append("  {:<20s} within-type sim: {}".format(dtype, val))
    lines.append("")
    if grouping["overall_across"] is not None:
        lines.append("  Across-type similarity:  {:.4f}".format(grouping["overall_across"]))
    if grouping["overall_within"] is not None:
        lines.append("  Within-type similarity:  {:.4f}".format(grouping["overall_within"]))
    if grouping["separation_ratio"] is not None:
        lines.append("  Separation ratio:        {:.2f}x".format(grouping["separation_ratio"]))
        lines.append("")
        if grouping["separation_ratio"] > 1.0:
            lines.append("  Interpretation: CLIP embeddings cluster by disaster type")
            lines.append("  (within-type similarity exceeds across-type similarity).")
        else:
            lines.append("  Interpretation: Disaster types are not strongly separated")
            lines.append("  in the CLIP embedding space.")
    lines.append("")

    # Section 5: Retrieval Transfer Matrix
    retrieval = analytics.cross_event_retrieval_matrix()
    lines.append("5. CROSS-EVENT RETRIEVAL TRANSFER MATRIX")
    lines.append("-" * 40)
    lines.append("  Rows = source event, Columns = target centroid")
    lines.append("  Values = mean cosine similarity of source images to target centroid")
    lines.append("")

    # Short labels for formatting
    short = [e[:12] for e in events]
    col_w = 13
    header_line = "  {:<15s}".format("") + "".join("{:>{w}s}".format(s, w=col_w) for s in short)
    lines.append(header_line)
    lines.append("  " + "-" * (15 + col_w * len(events)))
    for i, event in enumerate(events):
        row_vals = "".join("{:>{w}.4f}".format(float(retrieval[i, j]), w=col_w) for j in range(len(events)))
        lines.append("  {:<15s}{}".format(event[:15], row_vals))
    lines.append("")

    lines.append("=" * 68)
    lines.append("  Report generated by CLIP Embedding Analytics")
    lines.append("  Author: Rashid  |  Supervisor: XinHui Ma")
    lines.append("=" * 68)

    return "\n".join(lines)


def _generate_structured_json(analytics) -> dict:
    """Generate a well-structured, human-readable JSON report."""
    summary = analytics.global_summary()
    stats = analytics.per_event_stats()
    matrix = analytics.inter_event_similarity_matrix()
    retrieval = analytics.cross_event_retrieval_matrix()
    loo = analytics.loo_classification_accuracy()
    grouping = analytics.disaster_type_grouping_analysis()
    events = analytics.events
    type_groups = config.DISASTER_TYPE_GROUPS

    return {
        "_report_info": {
            "title": "CLIP Embedding Space Analytics Report",
            "project": "Visualising Natural Disaster Image Embeddings",
            "model": "CLIP ViT-B/32",
            "dataset": "CrisisMMD",
            "author": "Rashid",
            "supervisor": "XinHui Ma",
        },
        "global_summary": {
            "_description": "Overall statistics of the CLIP embedding space across all images.",
            "total_images": summary["total_images"],
            "embedding_dimensions": summary["embedding_dim"],
            "num_events": summary["num_events"],
            "pairwise_similarity": {
                "mean": round(summary["global_mean_similarity"], 4),
                "std": round(summary["global_std_similarity"], 4),
                "min": round(summary["global_min_similarity"], 4),
                "max": round(summary["global_max_similarity"], 4),
            }
        },
        "per_event_statistics": {
            "_description": "Per-event cohesion (mean pairwise cosine similarity) and spread (std). Higher cohesion means images within an event are more visually similar.",
            "events": {
                event: {
                    "disaster_type": type_groups.get(event, "Unknown"),
                    "image_count": stats[event]["count"],
                    "cohesion": round(stats[event]["cohesion"], 4),
                    "spread": round(stats[event]["spread"], 4),
                }
                for event in events
            }
        },
        "inter_event_similarity": {
            "_description": "Centroid-to-centroid cosine similarity between events. Symmetric matrix showing how similar each pair of events is in the embedding space.",
            "events": events,
            "matrix": [
                [round(float(matrix[i, j]), 4) for j in range(len(events))]
                for i in range(len(events))
            ]
        },
        "cross_disaster_transfer": {
            "_description": "Directional retrieval transfer matrix. matrix[i][j] = mean cosine similarity of source event i's images to target event j's centroid. NOT symmetric.",
            "events": events,
            "matrix": [
                [round(float(retrieval[i, j]), 4) for j in range(len(events))]
                for i in range(len(events))
            ]
        },
        "leave_one_out_classification": {
            "_description": "Each event is held out and its images are classified against remaining centroids by disaster TYPE match (not exact event). California Wildfires gets 0% because it is the only wildfire event.",
            "overall_accuracy": round(loo["overall_accuracy"], 4),
            "by_disaster_type": {
                t: round(a, 4) for t, a in sorted(loo["type_accuracies"].items())
            },
            "by_event": {
                event: round(loo["event_accuracies"][event], 4)
                for event in events
            }
        },
        "disaster_type_grouping": {
            "_description": "Tests whether CLIP representations cluster by disaster type. Separation ratio > 1.0 means within-type similarity exceeds across-type similarity.",
            "within_type_similarity": {
                t: round(v, 4) if v is not None else None
                for t, v in sorted(grouping["within_type"].items())
            },
            "overall_within_type": round(grouping["overall_within"], 4) if grouping["overall_within"] else None,
            "overall_across_type": round(grouping["overall_across"], 4) if grouping["overall_across"] else None,
            "separation_ratio": round(grouping["separation_ratio"], 2) if grouping["separation_ratio"] else None,
        }
    }


@app.callback(
    Output("analytics-download", "data"),
    Input("btn-export-json", "n_clicks"),
    Input("btn-export-txt", "n_clicks"),
    prevent_initial_call=True
)
def download_analytics_report(json_clicks, txt_clicks):
    """Generate and send the analytics report in the requested format."""
    trigger = ctx.triggered_id
    if trigger is None:
        return no_update

    analytics = get_analytics()

    if trigger == "btn-export-json":
        report = _generate_structured_json(analytics)
        return dcc.send_string(
            json.dumps(report, indent=2),
            filename="clip_analytics_report.json"
        )
    elif trigger == "btn-export-txt":
        text = _generate_text_report(analytics)
        return dcc.send_string(text, filename="clip_analytics_report.txt")

    return no_update


if __name__ == "__main__":
    print("\nServer running at http://127.0.0.1:8050/")
    app.run(debug=False)
