"""
CrisisMMD Demo Visualization App.

Interactive UMAP scatter plot with semantic search capability.
This is the frontend that brings together CLIP embeddings and UMAP reduction.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import json
import sys
from pathlib import Path


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Load the UMAP data
DATA_PATH = PROJECT_ROOT / "data" / "visualisation" / "umap_data.json"

try:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print("Loaded {} points from {}".format(len(df), DATA_PATH))
except FileNotFoundError:
    print("Error: '{}' not found.".format(DATA_PATH))
    print("Run umap_reduction.py first.")
    sys.exit(1)


# Initialize App
app = dash.Dash(__name__)


# Layout
app.layout = html.Div([
    
    # Header
    html.Div([
        html.H1(
            "CrisisMMD: Unsupervised Exploration",
            style={"color": "white", "fontFamily": "Arial"}
        ),
        html.H4(
            "Multimodal CLIP Embeddings + UMAP Projection",
            style={"color": "#888", "marginTop": "-10px"}
        )
    ], style={
        "backgroundColor": "#111",
        "padding": "20px",
        "textAlign": "center"
    }),

    # Main Grid
    html.Div([
        
        # Left: Controls
        html.Div([
            html.Label(
                "Semantic Search (Zero-Shot)",
                style={"color": "white", "fontWeight": "bold"}
            ),
            dcc.Input(
                id="search-input",
                type="text",
                placeholder='e.g., "flooded bridge" or "people in tents"',
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginTop": "10px"
                }
            ),
            html.Button(
                "Search",
                id="search-btn",
                n_clicks=0,
                style={
                    "width": "100%",
                    "marginTop": "10px",
                    "backgroundColor": "#007bff",
                    "color": "white",
                    "border": "none",
                    "padding": "10px",
                    "cursor": "pointer"
                }
            ),
            
            html.Hr(style={"borderColor": "#444"}),
            
            html.Div(id="click-data", style={"color": "white"})
            
        ], style={
            "width": "20%",
            "display": "inline-block",
            "verticalAlign": "top",
            "padding": "20px",
            "backgroundColor": "#222",
            "height": "80vh"
        }),

        # Right: The Map
        html.Div([
            dcc.Graph(id="umap-graph", style={"height": "80vh"})
        ], style={
            "width": "75%",
            "display": "inline-block",
            "verticalAlign": "top",
            "padding": "10px"
        })
        
    ], style={"display": "flex"})

], style={
    "backgroundColor": "#000",
    "height": "100vh",
    "margin": "0"
})


@app.callback(
    Output("umap-graph", "figure"),
    [Input("search-btn", "n_clicks")],
    [State("search-input", "value")]
)
def update_graph(n_clicks, search_term):
    """Update the scatter plot."""
    
    fig = go.Figure()

    # Add the main cluster points
    fig.add_trace(go.Scattergl(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(
            size=5,
            color="#00cc96",
            opacity=0.6
        ),
        text=df["path"],
        name="Images"
    ))

    # Dark mode styling
    fig.update_layout(
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan"
    )

    return fig


if __name__ == "__main__":
    print("Server running. Open http://127.0.0.1:8050/ in your browser.")
    app.run_server(debug=True)
