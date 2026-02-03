"""
CrisisMMD Visualisation App - Frontend.

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
import os
from pathlib import Path
from flask import send_from_directory

# Import backend functions
from app_backend import (
    get_manager,
    get_dataframe,
    get_unique_events,
    semantic_search,
    visual_search,
    classify_image,
    CLASSIFICATION_LABELS,
    LABEL_DISPLAY_NAMES,
    PROJECT_ROOT,
    IMAGE_FOLDER
)


# Initialise backend and get data
manager = get_manager()
df = get_dataframe()
UNIQUE_EVENTS = get_unique_events()


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
    score_pct = confidence * 100
    
    badge_class = "badge-academic"
    if confidence >= 0.28:
        badge_class += " badge-high"
    elif confidence >= 0.24:
        badge_class += " badge-mid"

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
                            "In the aftermath of a natural disaster, social media platforms become flooded with ",
                            "millions of images from affected areas. For humanitarian organisations, this presents ",
                            "a critical challenge: identifying relevant, actionable images from the deluge. ",
                            html.Strong("Traditional keyword-based search fails"), " because crisis images often lack ",
                            "descriptive text, and pre-defined labels cannot anticipate every possible disaster scenario."
                        ]),
                        html.P([
                            "This project demonstrates a ", html.Strong("zero-shot semantic search"), " approach using ",
                            "OpenAI's CLIP model. By encoding images into a shared embedding space with natural language, ",
                            "we enable responders to query the dataset using plain English—describing ",
                            html.Em("what they need to see"), " rather than relying on pre-existing tags."
                        ]),
                        html.P([
                            "The visualisation uses ", html.Strong("UMAP dimensionality reduction"), " to project the ",
                            "512-dimensional embedding space into an interactive 2D scatter plot. This reveals the ",
                            "semantic structure of the dataset, showing how images cluster by content type and disaster event."
                        ])
                    ], className="paper-card")
                ], md=8),
                dbc.Col([
                    html.Div([
                        html.H4("Technical Pipeline", className="paper-title mb-3"),
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
                html.Hr(className="my-4"),
                html.Div([
                    html.P("CrisisMMD Visualisation Tool", className="fw-bold mb-2"),
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
        ], fluid=True, style={"maxWidth": "1200px"}, className="py-4")
    ])


def explorer_page():
    """Interactive explorer page."""
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
        
        score_color = "#16a34a" if score >= 0.30 else "#2563eb"
        
        return html.Div([
            html.Img(src=img_url, className="w-100", 
                     style={"borderRadius": "4px 4px 0 0", "objectFit": "cover", "height": "120px"}),
            html.Div([
                html.Small(row["event"], className="text-secondary d-block"),
                html.Span(
                    "{}: {:.0f}%".format(score_label, score * 100),
                    style={"color": score_color, "fontWeight": "600"}
                ),
                html.Div(badge_elements, className="mt-2") if badge_elements else None
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
                    size=8, color="#2563eb", opacity=0.9,
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
    
    if not images and not query and clicked_index is None:
         image_grid = html.Div([
             html.P("Select an event or click a data point on the projection to view details.", 
                    className="text-muted small text-center mt-5")
         ])

    return fig, image_grid, status, gallery_title


if __name__ == "__main__":
    print("\nServer running at http://127.0.0.1:8050/")
    app.run(debug=False)
