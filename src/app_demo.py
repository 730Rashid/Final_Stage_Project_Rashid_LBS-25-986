"""
CrisisMMD Visualization App - Academic Professional Edition.

Clean, polished UI for academic presentations.
CLIP-powered semantic search with image gallery.

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
    q = clip_model.encode([query], convert_to_numpy=True)[0]
    q = q / np.linalg.norm(q)
    sims = embeddings_norm @ q
    idx = np.argsort(sims)[::-1][:top_k]
    return idx, sims[idx]


def parse_event(path):
    m = {"california_wildfires": "California Wildfires", "hurricane_harvey": "Hurricane Harvey", 
         "hurricane_irma": "Hurricane Irma", "hurricane_maria": "Hurricane Maria",
         "iraq_iran_earthquake": "Iraq-Iran Earthquake", "mexico_earthquake": "Mexico Earthquake",
         "srilanka_floods": "Sri Lanka Floods"}
    for k, v in m.items():
        if k in path.lower():
            return v
    return "Unknown"


df["event"] = df["path"].apply(parse_event)
df["hover"] = df.apply(lambda r: "<b>{}</b><br>{}".format(r["event"], Path(r["path"]).stem), axis=1)

# Refined academic color palette
COLORS = {
    "California Wildfires": "#B71C1C",
    "Hurricane Harvey": "#0D47A1", 
    "Hurricane Irma": "#1B5E20",
    "Hurricane Maria": "#4A148C",
    "Iraq-Iran Earthquake": "#E65100",
    "Mexico Earthquake": "#006064",
    "Sri Lanka Floods": "#3E2723",
    "Unknown": "#616161"
}


app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


@server.route("/images/<path:p>")
def serve_image(p):
    path = IMAGE_FOLDER / p
    return flask.send_file(str(path)) if path.exists() else ("Not found", 404)


def img_card(filepath, score):
    try:
        rel = Path(filepath).relative_to(IMAGE_FOLDER)
        src = "/images/{}".format(str(rel).replace("\\", "/"))
    except:
        src = "/images/{}".format(Path(filepath).name)
    
    event = parse_event(filepath)
    clr = COLORS.get(event, "#616161")
    
    return html.Div([
        html.Div([
            html.Img(src=src, style={
                "width": "100%", "height": "90px", "objectFit": "cover", "borderRadius": "4px"
            })
        ]),
        html.Div([
            html.Div(event, style={"fontSize": "11px", "color": "#424242", "fontWeight": "500", "marginBottom": "4px"}),
            html.Div([
                html.Span("Similarity: ", style={"color": "#757575", "fontSize": "10px"}),
                html.Span("{:.0%}".format(score), style={"color": clr, "fontWeight": "700", "fontSize": "13px"})
            ])
        ], style={"padding": "8px"})
    ], style={
        "backgroundColor": "white", "borderRadius": "8px", "overflow": "hidden",
        "border": "1px solid #E0E0E0", "transition": "box-shadow 0.2s"
    })


# ============ OVERVIEW PAGE ============
overview = html.Div([
    
    # 1. HEADER SECTION (Identity & Context)
    html.Div([
        html.H1("Unsupervised Multimodal Exploration", style={"color": "#00695C", "marginBottom": "10px", "fontWeight": "bold"}),
        html.H3("Honours Stage Project | Midway Technical Review", style={"color": "#555", "fontWeight": "normal", "marginTop": "0"}),
        html.P([
            html.Span("Student: ", style={"fontWeight": "bold", "color": "#333"}), "Rashid  |  ",
            html.Span("Dataset: ", style={"fontWeight": "bold", "color": "#333"}), "CrisisMMD (17,463 Images)"
        ], style={"color": "#666", "marginTop": "15px", "fontSize": "18px"})
    ], style={"marginBottom": "40px", "borderBottom": "1px solid #eee", "paddingBottom": "30px"}),

    # 2. THE PROBLEM & SOLUTION (Two Columns)
    html.Div([
        # Column 1: The Context
        html.Div([
            html.H4("The Challenge: Data Deluge", style={"color": "#d9534f", "borderLeft": "5px solid #d9534f", "paddingLeft": "15px", "fontSize": "22px"}),
            html.P("In the aftermath of a disaster, social media generates millions of unstructured images. Manual filtering is impossible for real-time situational awareness. Furthermore, traditional AI (CNNs) fails because it relies on rigid, supervised labels that cannot adapt to new, unseen disaster types.", style={"color": "#555", "lineHeight": "1.6"}),
        ], style={"width": "46%", "display": "inline-block", "verticalAlign": "top", "paddingRight": "4%"}),

        # Column 2: The Solution
        html.Div([
            html.H4("The Solution: Multimodal Learning", style={"color": "#00695C", "borderLeft": "5px solid #00695C", "paddingLeft": "15px", "fontSize": "22px"}),
            html.P("This tool implements an unsupervised 'Offline-Online' pipeline. It leverages the CLIP model (ViT-B/32) to extract 512-dimensional semantic vectors and UMAP to project the 'Global Structure' of the dataset into 2D, enabling Zero-Shot exploration.", style={"color": "#555", "lineHeight": "1.6"}),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"})
    ], style={"marginBottom": "50px"}),

    # 3. SYSTEM ARCHITECTURE (Visual Block)
    html.Div([
        html.H4("System Architecture Pipeline", style={"marginTop": "0", "color": "#333", "marginBottom": "20px"}),
        html.Div([
            # Step 1
            html.Div([
                html.Strong("1. Ingestion", style={"fontSize": "18px", "color": "#333"}), html.Br(),
                html.Span("Cleaning & Validation", style={"fontSize": "14px", "color": "#777"})
            ], style={"display": "inline-block", "width": "22%", "textAlign": "center", "borderRight": "1px solid #ddd", "padding": "10px"}),
            
            # Step 2
            html.Div([
                html.Strong("2. Vectorisation", style={"fontSize": "18px", "color": "#333"}), html.Br(),
                html.Span("CLIP Model (Offline)", style={"fontSize": "14px", "color": "#777"})
            ], style={"display": "inline-block", "width": "22%", "textAlign": "center", "borderRight": "1px solid #ddd", "padding": "10px"}),
            
            # Step 3
            html.Div([
                html.Strong("3. Projection", style={"fontSize": "18px", "color": "#333"}), html.Br(),
                html.Span("UMAP Reduction", style={"fontSize": "14px", "color": "#777"})
            ], style={"display": "inline-block", "width": "22%", "textAlign": "center", "borderRight": "1px solid #ddd", "padding": "10px"}),
            
            # Step 4
            html.Div([
                html.Strong("4. Interface", style={"fontSize": "18px", "color": "#00695C"}), html.Br(),
                html.Span("Dash / Plotly (Live)", style={"fontSize": "14px", "color": "#00695C", "fontWeight": "bold"})
            ], style={"display": "inline-block", "width": "22%", "textAlign": "center", "padding": "10px"}),
        ], style={"backgroundColor": "#f8f9fa", "padding": "30px", "borderRadius": "12px", "border": "1px solid #e9ecef", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"})
    ], style={"marginBottom": "50px"}),

    # 4. INSTRUCTIONS (Call to Action)
    html.Div([
        html.H3("Ready to Explore?", style={"color": "#333", "marginTop": "0"}),
        html.P("Navigate to the 'Explorer' tab to test the Zero-Shot Search capability.", style={"marginBottom": "15px"}),
        html.Ul([
            html.Li("Search for concepts like 'fire', 'rubble', or 'flooded houses'."),
            html.Li("The Map (Left) will highlight the global cluster structure."),
            html.Li("The Gallery (Right) will retrieve visual evidence based on Cosine Similarity.")
        ], style={"lineHeight": "1.8", "color": "#555"})
    ], style={"padding": "30px", "backgroundColor": "#e8f5e9", "borderRadius": "8px", "borderLeft": "6px solid #00695C"})

], style={"padding": "60px", "fontFamily": "Segoe UI, sans-serif", "maxWidth": "1200px", "margin": "0 auto", "backgroundColor": "#F5F6FA", "minHeight": "calc(100vh - 60px)"})


# ============ EXPLORER PAGE ============
explorer = html.Div([
    html.Div([
        # Sidebar
        html.Div([
            # Search Section
            html.Div([
                html.Div("Semantic Search", style={
                    "fontSize": "15px", "fontWeight": "600", "color": "#00695C", "marginBottom": "8px"
                }),
                html.Div("Query images using CLIP", style={
                    "fontSize": "11px", "color": "#757575", "marginBottom": "15px", "lineHeight": "1.5"
                }),
                dcc.Input(id="search-input", type="text", placeholder='e.g. "flooded houses"', debounce=True, style={
                    "width": "100%", "padding": "14px 16px", "border": "2px solid #E0E0E0", "borderRadius": "8px",
                    "fontSize": "14px", "boxSizing": "border-box", "backgroundColor": "#FAFAFA"
                }),
                html.Button("Search", id="search-btn", style={
                    "width": "100%", "marginTop": "12px", "padding": "14px", "backgroundColor": "#00695C",
                    "color": "white", "border": "none", "borderRadius": "8px", "cursor": "pointer",
                    "fontWeight": "600", "fontSize": "14px"
                }),
                html.Button("Reset View", id="clear-btn", style={
                    "width": "100%", "marginTop": "8px", "padding": "12px", "backgroundColor": "transparent",
                    "color": "#757575", "border": "1px solid #E0E0E0", "borderRadius": "8px", "cursor": "pointer"
                }),
                html.Div(id="search-result", style={
                    "marginTop": "15px", "padding": "12px", "backgroundColor": "#E8F5E9",
                    "borderRadius": "6px", "color": "#2E7D32", "fontSize": "13px", "display": "none"
                })
            ], style={"marginBottom": "25px"}),
            
            # Legend
            html.Div([
                html.Div("Event Legend", style={
                    "fontSize": "12px", "fontWeight": "600", "color": "#757575", "marginBottom": "12px",
                    "textTransform": "uppercase", "letterSpacing": "0.5px"
                }),
                html.Div([
                    html.Div([
                        html.Div(style={
                            "width": "12px", "height": "12px", "backgroundColor": c,
                            "borderRadius": "3px", "marginRight": "10px"
                        }),
                        html.Span(e, style={"fontSize": "12px", "color": "#424242"})
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})
                    for e, c in COLORS.items() if e != "Unknown"
                ])
            ])
            
        ], style={
            "width": "260px", "padding": "25px", "backgroundColor": "white",
            "borderRight": "1px solid #E8E8E8"
        }),
        
        # Main Content Area - Scatter Plot
        html.Div([
            # Plot Description Box
            html.Div([
                html.Div([
                    html.Strong("UMAP Embedding Projection", style={"fontSize": "15px", "color": "#00695C"}),
                    html.Span("  Global Structure Visualisation", style={"fontSize": "13px", "color": "#666"})
                ], style={"marginBottom": "8px"}),
                html.Div([
                    "Each dot = 1 image (17,463 total). ",
                    html.Strong("Close dots = similar images"), " based on CLIP embeddings (512 dimension, cosine distance)."
                ], style={"fontSize": "12px", "color": "#555", "lineHeight": "1.6"})
            ], style={
                "padding": "15px 20px", "backgroundColor": "#f8f9fa", 
                "borderBottom": "1px solid #E8E8E8", "borderLeft": "4px solid #00695C"
            }),
            
            # Graph Container
            html.Div([
                dcc.Graph(id="umap-graph", style={"height": "100%"}, config={
                    "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]
                })
            ], style={"height": "calc(100% - 100px)", "backgroundColor": "#FAFAFA"})
        ], style={"flex": "1", "display": "flex", "flexDirection": "column"}),
        
        # Gallery Panel
        html.Div([
            html.Div("Top Matches", style={
                "fontSize": "14px", "fontWeight": "600", "color": "#00695C", "marginBottom": "6px"
            }),
            html.Div("Sorted by CLIP cosine similarity score", style={
                "fontSize": "11px", "color": "#757575", "marginBottom": "15px"
            }),
            html.Div(id="image-gallery", style={
                "display": "grid", "gridTemplateColumns": "1fr", "gap": "12px",
                "overflowY": "auto", "maxHeight": "calc(100vh - 180px)"
            })
        ], style={
            "width": "250px", "padding": "25px", "backgroundColor": "white",
            "borderLeft": "1px solid #E8E8E8"
        })
        
    ], style={"display": "flex", "height": "calc(100vh - 60px)", "backgroundColor": "#F0F2F5"})
])


# Main Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Span("Rashid Disaster Research Project 2026", style={"fontSize": "18px", "fontWeight": "700", "color": "white"})
        ])
    ], style={
        "background": "linear-gradient(90deg, #00695C, #00897B)",
        "padding": "18px 30px"
    }),
    
    # Tabs
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Overview", value="overview", style={
            "padding": "14px 24px", "fontSize": "14px", "fontWeight": "500"
        }, selected_style={
            "padding": "14px 24px", "fontSize": "14px", "fontWeight": "600",
            "borderBottom": "3px solid #00695C", "color": "#00695C"
        }),
        dcc.Tab(label="Explorer", value="explorer", style={
            "padding": "14px 24px", "fontSize": "14px", "fontWeight": "500"
        }, selected_style={
            "padding": "14px 24px", "fontSize": "14px", "fontWeight": "600",
            "borderBottom": "3px solid #00695C", "color": "#00695C"
        })
    ], style={"backgroundColor": "white", "borderBottom": "1px solid #E0E0E0"}),
    
    html.Div(id="tab-content")
], style={"fontFamily": "'Segoe UI', 'Inter', -apple-system, sans-serif", "margin": "0"})


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render(tab):
    return overview if tab == "overview" else explorer


@app.callback(
    [Output("umap-graph", "figure"), Output("search-result", "children"), 
     Output("search-result", "style"), Output("image-gallery", "children")],
    [Input("search-btn", "n_clicks"), Input("search-input", "value"), Input("clear-btn", "n_clicks")]
)
def update(_, query, __):
    ctx = callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    
    fig = go.Figure()
    result = ""
    result_style = {"display": "none"}
    gallery = [html.P("Enter a search term to find similar disaster images", style={
        "color": "#9E9E9E", "fontSize": "13px", "textAlign": "center", "padding": "30px 10px"
    })]
    hl = set()
    paths, scores = [], []
    
    if query and trig in ["search-btn", "search-input"] and len(query.strip()) > 2:
        idx, sims = semantic_search(query.strip())
        hl = set(idx)
        paths = [df.iloc[i]["path"] for i in idx[:10]]
        scores = sims[:10]
        result = "Found {} matching images".format(len(idx))
        result_style = {
            "marginTop": "15px", "padding": "12px", "backgroundColor": "#E8F5E9",
            "borderRadius": "6px", "color": "#2E7D32", "fontSize": "13px"
        }
    
    if hl:
        mask = df.index.isin(hl)
        fig.add_trace(go.Scattergl(
            x=df[~mask]["x"], y=df[~mask]["y"], mode="markers",
            marker=dict(size=4, color="#E0E0E0", opacity=0.4),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scattergl(
            x=df[mask]["x"], y=df[mask]["y"], mode="markers",
            marker=dict(size=10, color="#E65100", opacity=1, line=dict(width=2, color="white")),
            text=df[mask]["hover"], hovertemplate="%{text}<extra></extra>", name="Matches"
        ))
    else:
        for e in df["event"].unique():
            ev = df[df["event"] == e]
            fig.add_trace(go.Scattergl(
                x=ev["x"], y=ev["y"], mode="markers",
                marker=dict(size=6, color=COLORS.get(e, "#616161"), opacity=0.8),
                text=ev["hover"], hovertemplate="%{text}<extra></extra>", name=e
            ))
    
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="#FAFAFA",
        font=dict(family="Segoe UI, sans-serif", color="#424242"),
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False, showticklabels=False, title=""),
        margin=dict(l=15, r=15, t=40, b=15),
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        dragmode="pan"
    )
    
    if paths:
        gallery = [img_card(p, float(s)) for p, s in zip(paths, scores)]
    
    return fig, result, result_style, gallery


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  CrisisMMD Visualisation Server")
    print("  http://127.0.0.1:8050/")
    print("="*50 + "\n")
    app.run(debug=False, use_reloader=False)
