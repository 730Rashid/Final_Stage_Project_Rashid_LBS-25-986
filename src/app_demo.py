"""
CrisisMMD Visualisation App with Zero Shot Classification.

Real time CLIP semantic search with UMAP visualisation and automatic
image tagging using zero shot classification.

Features:
    Event Type Filtering: Filter by disaster event before searching.
    Semantic Search: Natural language queries using CLIP.
    Visual Query: Click on any image to find visually similar images.
    Zero Shot Tags: Automatic classification of image content.
    UMAP Visualisation: Interactive 2D embedding map.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import dash
from dash import dcc, html, Input, Output, State, ctx
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

print("[System] Initialising Application...")


# Load Metadata
try:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except FileNotFoundError:
    print("[Error] Could not find {}. Run umap_reduction.py first.".format(DATA_PATH))
    exit()


def parse_event(path):
    """Extract event name from folder structure for cleaner tooltips."""
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
print("Metadata loaded: {} images across {} events".format(
    len(df), len(UNIQUE_EVENTS)
))


# Load CLIP Model
print("Loading CLIP Model (ViT-B/32)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Inference Device: {}".format(device))

try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading Image Embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print("Embeddings loaded. Shape: {}".format(embeddings.shape))
except Exception as e:
    print("Failed to load AI models: {}".format(e))
    exit()


# Zero Shot Classification Labels
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

# Precompute label embeddings
print("[Status] Precomputing label embeddings for zero shot classification...")
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
    
    print("[Status] Label embeddings ready. Shape: {}".format(label_embeddings.shape))
except Exception as e:
    print("[Warning] Could not precompute label embeddings: {}".format(e))


# Search Functions

def semantic_search(query, subset_indices=None, top_k=50):
    """
    Encode a text query and find the most similar images.
    
    Args:
        query: Natural language search query.
        subset_indices: Optional list of indices to search within.
        top_k: Number of results to return.
        
    Returns:
        Tuple of (indices into dataframe, similarity scores).
    """
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
        top_scores = similarities[local_top_indices]
        
        return global_indices, top_scores
    else:
        similarities = cosine_similarity(text_vector, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices, similarities[top_indices]


def visual_search(image_index, subset_indices=None, top_k=50):
    """
    Find images visually similar to the selected image.
    
    Args:
        image_index: Index of the query image in the embeddings array.
        subset_indices: Optional list of indices to search within.
        top_k: Number of results to return.
        
    Returns:
        Tuple of (indices into dataframe, similarity scores).
    """
    query_vector = embeddings[image_index].reshape(1, -1)
    
    if subset_indices is not None and len(subset_indices) > 0:
        subset_embeddings = embeddings[subset_indices]
        similarities = cosine_similarity(query_vector, subset_embeddings)[0]
        
        local_top_k = min(top_k, len(subset_indices))
        local_top_indices = np.argsort(similarities)[::-1][:local_top_k]
        
        global_indices = np.array(subset_indices)[local_top_indices]
        top_scores = similarities[local_top_indices]
        
        return global_indices, top_scores
    else:
        similarities = cosine_similarity(query_vector, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices, similarities[top_indices]


def classify_image(image_index, threshold=0.20):
    """
    Classify an image using zero shot classification.
    
    Compares the image embedding against precomputed label embeddings
    to determine what concepts are present in the image.
    
    Args:
        image_index: Index of the image in the embeddings array.
        threshold: Minimum similarity score to include a label.
        
    Returns:
        List of tuples containing (label_name, confidence_score).
    """
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


def create_badge(label, confidence):
    """
    Create an HTML badge element for a classification label.
    
    Args:
        label: The display name of the label.
        confidence: Confidence score between 0 and 1.
        
    Returns:
        A Dash HTML component representing the badge.
    """
    if confidence >= 0.30:
        bg_colour = "#2e7d32"
    elif confidence >= 0.25:
        bg_colour = "#1976d2"
    else:
        bg_colour = "#757575"
    
    return html.Span(
        "{} ({:.0f}%)".format(label, confidence * 100),
        style={
            "backgroundColor": bg_colour,
            "color": "white",
            "padding": "2px 6px",
            "borderRadius": "3px",
            "fontSize": "9px",
            "marginRight": "3px",
            "display": "inline-block",
            "marginBottom": "2px"
        }
    )


# Flask Server
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


@server.route("/images/<path:p>")
def serve_image(p):
    """Serve images directly from the data folder."""
    path = IMAGE_FOLDER / p
    if path.exists():
        return send_from_directory(str(path.parent), path.name)
    return "Not found", 404


# Styles
NAV_STYLE = {
    "display": "flex",
    "borderBottom": "1px solid #ddd",
    "padding": "15px 40px",
    "backgroundColor": "white",
    "alignItems": "center",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"
}

LINK_STYLE = {
    "marginRight": "30px",
    "textDecoration": "none",
    "color": "#005a8c",
    "fontWeight": "bold",
    "fontSize": "16px",
    "fontFamily": "Segoe UI"
}

BUTTON_STYLE = {
    "padding": "10px 25px",
    "marginLeft": "10px",
    "backgroundColor": "#005a8c",
    "color": "white",
    "border": "none",
    "borderRadius": "4px",
    "cursor": "pointer"
}

BUTTON_SECONDARY_STYLE = {
    "padding": "8px 15px",
    "marginLeft": "10px",
    "backgroundColor": "#6c757d",
    "color": "white",
    "border": "none",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontSize": "12px"
}


# Page Layouts

def overview_page():
    """Create the overview landing page."""
    return html.Div([
        html.Div([
            html.H1(
                "Unsupervised Multimodal Exploration",
                style={"color": "#005a8c", "fontWeight": "bold"}
            ),
            html.H3(
                "Honours Stage Project",
                style={"color": "#555", "fontWeight": "normal"}
            ),
            html.P([
                html.Span("Student: ", style={"fontWeight": "bold"}),
                "Rashid  |  ",
                html.Span("Dataset: ", style={"fontWeight": "bold"}),
                "CrisisMMD (17,463 Images)"
            ], style={"color": "#666", "fontSize": "18px"})
        ], style={
            "borderBottom": "1px solid #eee",
            "paddingBottom": "30px",
            "marginBottom": "40px"
        }),

        html.Div([
            html.H4("System Architecture", style={"color": "#333"}),
            html.Div([
                html.Strong("1. Ingestion"), " then ",
                html.Strong("2. Vectorisation (CLIP)"), " then ",
                html.Strong("3. Projection (UMAP)"), " then ",
                html.Strong("4. Interface (Dash)", style={"color": "#005a8c"})
            ], style={
                "backgroundColor": "#f8f9fa",
                "padding": "20px",
                "borderRadius": "8px",
                "textAlign": "center"
            })
        ]),
        
        html.Div([
            html.H4("Zero Shot Classification Labels", style={"color": "#333", "marginTop": "30px"}),
            html.P(
                "The system can automatically detect the following concepts in images:",
                style={"color": "#666"}
            ),
            html.Div([
                create_badge(LABEL_DISPLAY_NAMES[label], 0.30) 
                for label in CLASSIFICATION_LABELS
            ], style={"marginTop": "10px"})
        ])
    ], style={
        "padding": "60px",
        "maxWidth": "1200px",
        "margin": "0 auto",
        "fontFamily": "Segoe UI"
    })


def explorer_page():
    """Create the interactive explorer page with event filtering and visual query."""
    return html.Div([
        html.Div([
            html.Div([
                html.Label(
                    "Filter by Event:",
                    style={"fontWeight": "bold", "marginRight": "10px"}
                ),
                dcc.Dropdown(
                    id="event-filter",
                    options=[{"label": "All Events", "value": "all"}] +
                            [{"label": e, "value": e} for e in UNIQUE_EVENTS],
                    value="all",
                    clearable=False,
                    style={
                        "width": "200px",
                        "display": "inline-block",
                        "verticalAlign": "middle"
                    }
                )
            ], style={"display": "inline-block", "marginRight": "30px"}),

            html.Span(
                "|",
                style={"color": "#ccc", "marginRight": "30px", "fontSize": "24px"}
            ),

            html.Label(
                "Semantic Search:",
                style={"fontWeight": "bold", "marginRight": "10px"}
            ),
            dcc.Input(
                id="search-input",
                type="text",
                placeholder='e.g. flooded house',
                debounce=True,
                style={
                    "padding": "10px",
                    "width": "250px",
                    "borderRadius": "4px",
                    "border": "1px solid #ccc"
                }
            ),
            html.Button("Search", id="search-btn", n_clicks=0, style=BUTTON_STYLE),
            
            html.Button(
                "Clear Selection",
                id="clear-btn",
                n_clicks=0,
                style=BUTTON_SECONDARY_STYLE
            ),
            
            html.Div(
                id="search-status",
                style={
                    "display": "inline-block",
                    "marginLeft": "15px",
                    "color": "#666"
                }
            )
        ], style={
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "borderBottom": "1px solid #ddd",
            "textAlign": "center"
        }),

        html.Div(
            "Tip: Click on any point in the map to find visually similar images. "
            "Each result shows automatic content tags.",
            id="visual-query-hint",
            style={
                "padding": "8px",
                "backgroundColor": "#e8f4f8",
                "textAlign": "center",
                "fontSize": "13px",
                "color": "#005a8c",
                "borderBottom": "1px solid #ccc"
            }
        ),

        dcc.Store(id="clicked-point-store", data=None),

        html.Div([
            html.Div([
                html.H4(
                    "Global Structure (UMAP)",
                    style={"textAlign": "center", "color": "#555"}
                ),
                dcc.Graph(
                    id="umap-graph",
                    style={"height": "70vh"},
                    config={"displaylogo": False}
                )
            ], style={
                "width": "58%",
                "display": "inline-block",
                "verticalAlign": "top"
            }),

            html.Div([
                html.H4(
                    id="gallery-title",
                    children="Visual Evidence (Top 9)",
                    style={"textAlign": "center", "color": "#555"}
                ),
                html.Div(
                    id="image-grid",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(3, 1fr)",
                        "gap": "10px",
                        "padding": "10px",
                        "height": "70vh",
                        "overflowY": "auto"
                    }
                )
            ], style={
                "width": "40%",
                "display": "inline-block",
                "verticalAlign": "top"
            })
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


# Callbacks

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page(pathname):
    """Route to the correct page."""
    if pathname == "/explorer":
        return explorer_page()
    return overview_page()


@app.callback(
    Output("clicked-point-store", "data"),
    [Input("umap-graph", "clickData"), Input("clear-btn", "n_clicks")],
    prevent_initial_call=True
)
def handle_click(click_data, clear_clicks):
    """Store the clicked point index or clear it."""
    triggered_id = ctx.triggered_id
    
    if triggered_id == "clear-btn":
        return None
    
    if click_data and "points" in click_data:
        point = click_data["points"][0]
        if "pointIndex" in point:
            return point["pointIndex"]
    
    return None


def build_image_card(row, score, score_label="Score", score_colour="#005a8c"):
    """
    Build an image card with classification badges.
    
    Args:
        row: DataFrame row containing image metadata.
        score: Similarity or relevance score.
        score_label: Label for the score display.
        score_colour: Colour for the score text.
        
    Returns:
        A Dash HTML component representing the image card.
    """
    try:
        rel_path = Path(row["path"]).relative_to(IMAGE_FOLDER)
        img_url = "/images/{}".format(str(rel_path).replace(os.sep, "/"))
        
        image_idx = row["original_idx"]
        classifications = classify_image(image_idx, threshold=0.20)
        
        badge_elements = []
        for label, confidence in classifications[:3]:
            badge_elements.append(create_badge(label, confidence))
        
        return html.Div([
            html.Img(
                src=img_url,
                style={
                    "width": "100%",
                    "borderRadius": "4px",
                    "border": "1px solid #ddd"
                }
            ),
            html.Div([
                html.Div(
                    row["event"],
                    style={"fontSize": "10px", "color": "#888"}
                ),
                html.Div(
                    "{}: {:.1f}%".format(score_label, score * 100),
                    style={
                        "fontSize": "11px",
                        "color": score_colour,
                        "fontWeight": "bold"
                    }
                ),
                html.Div(
                    badge_elements,
                    style={"marginTop": "4px"}
                ) if badge_elements else None
            ], style={"textAlign": "center"})
        ])
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
    """
    Update the UMAP plot and image gallery.
    
    Handles three modes:
        1. Event filter only: Ghost non selected events.
        2. Text search: Find images matching the query.
        3. Visual query: Find images similar to the clicked point.
    """
    fig = go.Figure()
    images = []
    status = "Ready"
    gallery_title = "Visual Evidence (Top 9)"
    
    if selected_event and selected_event != "all":
        event_mask = df["event"] == selected_event
        filtered_df = df[event_mask]
        ghosted_df = df[~event_mask]
        filtered_indices = filtered_df["original_idx"].tolist()
        status = "Showing {:,} images from {}".format(
            len(filtered_df), selected_event
        )
    else:
        filtered_df = df
        ghosted_df = pd.DataFrame()
        filtered_indices = None
        status = "Showing all {:,} images".format(len(df))
    
    if len(ghosted_df) > 0:
        fig.add_trace(go.Scattergl(
            x=ghosted_df["x"],
            y=ghosted_df["y"],
            mode="markers",
            marker=dict(size=4, color="#d0d0d0", opacity=0.15),
            hoverinfo="skip",
            showlegend=False,
            name="Other Events"
        ))
    
    fig.add_trace(go.Scattergl(
        x=filtered_df["x"],
        y=filtered_df["y"],
        mode="markers",
        marker=dict(size=5, color="#a0c4e8", opacity=0.4),
        text=filtered_df["hover"],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        name="Active"
    ))

    if clicked_index is not None:
        indices, scores = visual_search(
            clicked_index,
            subset_indices=filtered_indices
        )
        match_df = df.iloc[indices].copy()
        match_df["score"] = scores
        
        query_row = df.iloc[clicked_index]
        fig.add_trace(go.Scattergl(
            x=[query_row["x"]],
            y=[query_row["y"]],
            mode="markers",
            marker=dict(
                size=18,
                color="#e53935",
                opacity=1.0,
                line=dict(width=2, color="white"),
                symbol="star"
            ),
            text=["Query Image"],
            hovertemplate="<b>Query Image</b><extra></extra>",
            showlegend=False,
            name="Query"
        ))
        
        fig.add_trace(go.Scattergl(
            x=match_df["x"],
            y=match_df["y"],
            mode="markers",
            marker=dict(
                size=10,
                color="#43a047",
                opacity=0.9,
                line=dict(width=1, color="white")
            ),
            text=match_df["hover"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="Similar"
        ))
        
        status = "Visual Query: Found {} similar images".format(len(indices))
        gallery_title = "Visually Similar Images"
        
        for _, row in match_df.head(9).iterrows():
            card = build_image_card(row, row["score"], "Similarity", "#43a047")
            if card:
                images.append(card)

    elif query and len(query.strip()) > 2:
        indices, scores = semantic_search(query.strip(), subset_indices=filtered_indices)
        match_df = df.iloc[indices].copy()
        match_df["score"] = scores
        
        fig.add_trace(go.Scattergl(
            x=match_df["x"],
            y=match_df["y"],
            mode="markers",
            marker=dict(
                size=12,
                color="#1976D2",
                opacity=1.0,
                line=dict(width=1.5, color="white")
            ),
            text=match_df["hover"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="Matches"
        ))
        
        if selected_event and selected_event != "all":
            status = "Found {} matches for '{}' in {}".format(
                len(indices), query, selected_event
            )
        else:
            status = "Found {} matches for '{}'".format(len(indices), query)
        
        gallery_title = "Search Results (Top 9)"
        
        for _, row in match_df.head(9).iterrows():
            card = build_image_card(row, row["score"], "Score", "#005a8c")
            if card:
                images.append(card)

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        dragmode="pan"
    )
    
    return fig, images, status, gallery_title


if __name__ == "__main__":
    print("\n[Server] Running at http://127.0.0.1:8050/")
    app.run(debug=False)
