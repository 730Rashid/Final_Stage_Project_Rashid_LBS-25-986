"""
CrisisMMD Visualization App - Professional Edition.

Polished UI with CLIP-powered semantic search.
Designed for academic presentations and supervisor demos.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import sys
import flask
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Configuration
DATA_PATH = PROJECT_ROOT / "data" / "visualisation" / "umap_data.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.npy"
IMAGE_FOLDER = PROJECT_ROOT / "data" / "processed" / "clean_data"


# Load data
print("Loading data...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)
embeddings = np.load(EMBEDDINGS_PATH)
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
print("Loaded {} points".format(len(df)))


# Load CLIP
print("Loading CLIP model...")
from sentence_transformers import SentenceTransformer
clip_model = SentenceTransformer("clip-ViT-B-32")
print("CLIP model ready!")


def semantic_search(query, top_k=100):
    query_embedding = clip_model.encode([query], convert_to_numpy=True)[0]
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    similarities = embeddings_norm @ query_norm
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, similarities[top_indices]


def parse_event(filepath):
    mapping = {
        "california_wildfires": "California Wildfires",
        "hurricane_harvey": "Hurricane Harvey",
        "hurricane_irma": "Hurricane Irma",
        "hurricane_maria": "Hurricane Maria",
        "iraq_iran_earthquake": "Iraq-Iran Earthquake",
        "mexico_earthquake": "Mexico Earthquake",
        "srilanka_floods": "Sri Lanka Floods"
    }
    for key, name in mapping.items():
        if key in filepath.lower():
            return name
    return "Unknown"


df["event"] = df["path"].apply(parse_event)
df["image_id"] = df["path"].apply(lambda x: Path(x).stem)
df["hover"] = df.apply(lambda r: "<b>{}</b><br>{}".format(r["event"], r["image_id"]), axis=1)


# Professional colour palette
COLORS = {
    "California Wildfires": "#C62828",
    "Hurricane Harvey": "#1565C0",
    "Hurricane Irma": "#2E7D32",
    "Hurricane Maria": "#6A1B9A",
    "Iraq-Iran Earthquake": "#EF6C00",
    "Mexico Earthquake": "#00838F",
    "Sri Lanka Floods": "#4E342E",
    "Unknown": "#78909C"
}


# App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


@server.route("/images/<path:p>")
def serve_image(p):
    path = IMAGE_FOLDER / p
    return flask.send_file(str(path)) if path.exists() else ("Not found", 404)


def image_card(filepath, score):
    try:
        rel = Path(filepath).relative_to(IMAGE_FOLDER)
        src = "/images/{}".format(str(rel).replace("\\", "/"))
    except:
        src = "/images/{}".format(Path(filepath).name)
    
    event = parse_event(filepath)
    color = COLORS.get(event, "#78909C")
    
    return html.Div([
        html.Div(style={
            "position": "absolute", "top": "0", "left": "0", "right": "0",
            "height": "4px", "backgroundColor": color, "borderRadius": "6px 6px 0 0"
        }),
        html.Img(src=src, style={
            "width": "100%", "height": "100px", "objectFit": "cover",
            "borderRadius": "0 0 4px 4px"
        }),
        html.Div([
            html.Span(event[:15], style={"fontSize": "11px", "fontWeight": "500", "color": "#37474F"}),
            html.Span("{:.0%}".format(score), style={"fontSize": "11px", "color": color, "fontWeight": "600"})
        ], style={"display": "flex", "justifyContent": "space-between", "padding": "6px 0 0"})
    ], style={
        "position": "relative", "backgroundColor": "white", "borderRadius": "6px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "overflow": "hidden", "padding": "0 0 6px"
    })


# Overview Page
overview = html.Div([
    html.Div([
        # Hero section
        html.Div([
            html.H2("Multimodal Disaster Image Analysis", style={
                "color": "#1A237E", "margin": "0 0 8px", "fontSize": "28px", "fontWeight": "700"
            }),
            html.P("Exploring 17,463 disaster images using CLIP embeddings and UMAP projection", style={
                "color": "#546E7A", "fontSize": "15px", "margin": "0"
            })
        ], style={"textAlign": "center", "marginBottom": "35px"}),
        
        # Stats cards
        html.Div([
            html.Div([
                html.Div("17,463", style={"fontSize": "32px", "fontWeight": "700", "color": "#1A237E"}),
                html.Div("Total Images", style={"fontSize": "12px", "color": "#78909C", "textTransform": "uppercase"})
            ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "#E8EAF6", "borderRadius": "8px", "flex": "1"}),
            html.Div([
                html.Div("512", style={"fontSize": "32px", "fontWeight": "700", "color": "#1A237E"}),
                html.Div("Embedding Dim", style={"fontSize": "12px", "color": "#78909C", "textTransform": "uppercase"})
            ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "#E8EAF6", "borderRadius": "8px", "flex": "1"}),
            html.Div([
                html.Div("7", style={"fontSize": "32px", "fontWeight": "700", "color": "#1A237E"}),
                html.Div("Disaster Events", style={"fontSize": "12px", "color": "#78909C", "textTransform": "uppercase"})
            ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "#E8EAF6", "borderRadius": "8px", "flex": "1"})
        ], style={"display": "flex", "gap": "15px", "marginBottom": "35px"}),
        
        # Architecture
        html.Div([
            html.H4("Technical Architecture", style={"color": "#37474F", "marginBottom": "15px", "fontSize": "16px"}),
            html.Div([
                html.Div([html.Strong("Image Encoder"), html.Br(), "CLIP ViT-B/32"], style={"padding": "12px", "backgroundColor": "#FAFAFA", "borderRadius": "6px", "textAlign": "center", "flex": "1", "fontSize": "13px"}),
                html.Div("→", style={"padding": "12px", "color": "#90A4AE"}),
                html.Div([html.Strong("Reduction"), html.Br(), "UMAP (cosine)"], style={"padding": "12px", "backgroundColor": "#FAFAFA", "borderRadius": "6px", "textAlign": "center", "flex": "1", "fontSize": "13px"}),
                html.Div("→", style={"padding": "12px", "color": "#90A4AE"}),
                html.Div([html.Strong("Search"), html.Br(), "Cosine Similarity"], style={"padding": "12px", "backgroundColor": "#FAFAFA", "borderRadius": "6px", "textAlign": "center", "flex": "1", "fontSize": "13px"})
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center"})
        ], style={"marginBottom": "25px"}),
        
        html.Div([
            html.P("Navigate to the Explorer tab to search for images using natural language.", style={"color": "#78909C", "textAlign": "center", "fontSize": "13px"})
        ])
        
    ], style={
        "maxWidth": "650px", "margin": "0 auto", "padding": "45px",
        "backgroundColor": "white", "borderRadius": "12px",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.06)"
    })
], style={"padding": "40px 20px", "backgroundColor": "#F5F7FA", "minHeight": "calc(100vh - 100px)"})


# Explorer Page
explorer = html.Div([
    html.Div([
        # Sidebar
        html.Div([
            html.Div([
                html.H4("Semantic Search", style={"margin": "0 0 5px", "fontSize": "15px", "color": "#1A237E", "fontWeight": "600"}),
                html.P("Find images by description", style={"margin": "0", "fontSize": "11px", "color": "#90A4AE"})
            ], style={"marginBottom": "15px"}),
            
            dcc.Input(id="search-input", type="text", placeholder='e.g. "flooded houses"', debounce=True, style={
                "width": "100%", "padding": "12px", "border": "2px solid #E0E0E0", "borderRadius": "8px",
                "fontSize": "13px", "boxSizing": "border-box", "outline": "none"
            }),
            
            html.Button("Search", id="search-btn", style={
                "width": "100%", "marginTop": "10px", "padding": "12px", "backgroundColor": "#1A237E",
                "color": "white", "border": "none", "borderRadius": "8px", "cursor": "pointer",
                "fontWeight": "600", "fontSize": "13px", "transition": "all 0.2s"
            }),
            
            html.Button("Clear", id="clear-btn", style={
                "width": "100%", "marginTop": "8px", "padding": "10px", "backgroundColor": "#ECEFF1",
                "color": "#546E7A", "border": "none", "borderRadius": "8px", "cursor": "pointer", "fontSize": "12px"
            }),
            
            html.Div(id="search-result", style={
                "marginTop": "15px", "padding": "10px", "backgroundColor": "#E8F5E9",
                "borderRadius": "6px", "fontSize": "12px", "color": "#2E7D32", "display": "none"
            }),
            
            html.Hr(style={"margin": "20px 0", "border": "none", "borderTop": "1px solid #E0E0E0"}),
            
            html.Div([
                html.Div("Event Legend", style={"fontSize": "11px", "color": "#90A4AE", "marginBottom": "10px", "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                html.Div([
                    html.Div([
                        html.Span(style={"display": "inline-block", "width": "10px", "height": "10px", "backgroundColor": c, "borderRadius": "3px", "marginRight": "8px"}),
                        html.Span(e, style={"fontSize": "11px", "color": "#546E7A"})
                    ], style={"marginBottom": "6px", "display": "flex", "alignItems": "center"})
                    for e, c in COLORS.items() if e != "Unknown"
                ])
            ])
        ], style={
            "width": "220px", "padding": "20px", "backgroundColor": "white",
            "borderRight": "1px solid #E0E0E0"
        }),
        
        # Graph
        html.Div([
            dcc.Graph(id="umap-graph", style={"height": "100%"}, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        ], style={"flex": "1", "padding": "15px", "backgroundColor": "#FAFAFA"}),
        
        # Gallery
        html.Div([
            html.Div([
                html.Span("Image Results", style={"fontSize": "14px", "fontWeight": "600", "color": "#1A237E"}),
            ], style={"marginBottom": "15px"}),
            html.Div(id="image-gallery", style={
                "display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px",
                "overflowY": "auto", "maxHeight": "calc(100vh - 180px)", "paddingRight": "5px"
            })
        ], style={"width": "280px", "padding": "20px", "backgroundColor": "white", "borderLeft": "1px solid #E0E0E0"})
        
    ], style={"display": "flex", "height": "calc(100vh - 60px)"})
])


# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("CrisisMMD", style={
                "margin": "0", "fontSize": "22px", "fontWeight": "700", "color": "white", "letterSpacing": "-0.5px"
            }),
            html.Span("Multimodal Disaster Analysis", style={
                "fontSize": "12px", "color": "#B3E5FC", "marginLeft": "12px", "fontWeight": "400"
            })
        ], style={"display": "flex", "alignItems": "baseline"})
    ], style={
        "background": "linear-gradient(135deg, #1A237E 0%, #283593 100%)",
        "padding": "16px 25px", "boxShadow": "0 2px 10px rgba(0,0,0,0.1)"
    }),
    
    # Tabs
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview", style={"padding": "12px 20px", "fontSize": "13px"}, selected_style={"padding": "12px 20px", "fontSize": "13px", "borderTop": "3px solid #1A237E"}),
        dcc.Tab(label="Explorer", value="explorer", style={"padding": "12px 20px", "fontSize": "13px"}, selected_style={"padding": "12px 20px", "fontSize": "13px", "borderTop": "3px solid #1A237E"})
    ], style={"backgroundColor": "white", "borderBottom": "1px solid #E0E0E0"}),
    
    html.Div(id="tab-content")
], style={"backgroundColor": "#F5F7FA", "minHeight": "100vh", "fontFamily": "'Inter', 'Segoe UI', -apple-system, sans-serif"})


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return overview if tab == "overview" else explorer


@app.callback(
    [Output("umap-graph", "figure"), Output("search-result", "children"), Output("search-result", "style"), Output("image-gallery", "children")],
    [Input("search-btn", "n_clicks"), Input("search-input", "value"), Input("clear-btn", "n_clicks")],
    prevent_initial_call=False
)
def update(_, query, __):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    
    fig = go.Figure()
    result_text = ""
    result_style = {"display": "none"}
    gallery = [html.P("Search for images to see results here", style={"color": "#90A4AE", "fontSize": "12px", "gridColumn": "span 2", "textAlign": "center", "padding": "20px 0"})]
    highlight = set()
    paths, scores = [], []
    
    if query and triggered in ["search-btn", "search-input"] and len(query.strip()) > 2:
        indices, sims = semantic_search(query.strip())
        highlight = set(indices)
        paths = [df.iloc[i]["path"] for i in indices[:12]]
        scores = sims[:12]
        result_text = "✓ Found {} matches".format(len(indices))
        result_style = {"marginTop": "15px", "padding": "10px", "backgroundColor": "#E8F5E9", "borderRadius": "6px", "fontSize": "12px", "color": "#2E7D32"}
    
    if highlight:
        mask = df.index.isin(highlight)
        fig.add_trace(go.Scattergl(x=df[~mask]["x"], y=df[~mask]["y"], mode="markers", marker=dict(size=4, color="#E0E0E0", opacity=0.25), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattergl(x=df[mask]["x"], y=df[mask]["y"], mode="markers", marker=dict(size=9, color="#FF5722", opacity=0.9, line=dict(width=1, color="white")), text=df[mask]["hover"], hovertemplate="%{text}<extra></extra>", name="Results"))
    else:
        for e in df["event"].unique():
            ev = df[df["event"] == e]
            fig.add_trace(go.Scattergl(x=ev["x"], y=ev["y"], mode="markers", marker=dict(size=5, color=COLORS.get(e, "#78909C"), opacity=0.7), text=ev["hover"], hovertemplate="%{text}<extra></extra>", name=e))
    
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="#FAFAFA",
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False, showticklabels=False, title=""),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.08, font=dict(size=10)),
        dragmode="pan", font=dict(family="Inter, sans-serif")
    )
    
    if paths:
        gallery = [image_card(p, float(s)) for p, s in zip(paths, scores)]
    
    return fig, result_text, result_style, gallery


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  CrisisMMD Visualisation Server")
    print("  http://127.0.0.1:8050/")
    print("="*50 + "\n")
    app.run(debug=False, use_reloader=False)
