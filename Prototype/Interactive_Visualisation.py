"""
CRISIS IMAGE EXPLORER PROTOTYPE (V3 - GRID LAYOUT)
==================================================
A clean, grid-based dashboard to demonstrate dataset loading and filtering.
Focuses on clarity, UI design, and data inspection.

Author: RASHID
Module: Honours Stage Project
Date: Oct 2025
"""

import dash
from dash import dcc, html, Input, Output, ALL, callback_context
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import random
import pandas as pd

# ============================================================================
# ⚙️ CONFIGURATION
# ============================================================================
CRISISMD_PATH = r"Prototype/data_image" 
MAX_IMAGES_TO_LOAD = 50  # Keeping it small for a fast, snappy demo

# ============================================================================
# 🔧 HELPER FUNCTIONS
# ============================================================================

def load_data(path_str):
    """Scans directory and creates a simple dataframe of images."""
    data_path = Path(path_str)
    
    exts = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in exts:
        files.extend(list(data_path.rglob(ext)))
    
    if not files:
        return pd.DataFrame()

    # Shuffle and slice
    random.shuffle(files)
    selected_files = files[:MAX_IMAGES_TO_LOAD]
    print(f"✓ Loaded {len(selected_files)} images for grid view.")

    images = []
    for i, img_path in enumerate(selected_files):
        # Extract category from folder name
        try:
            parts = img_path.relative_to(data_path).parts
            category = parts[0] if len(parts) > 1 else "Uncategorised"
        except ValueError:
            category = "Uncategorised"

        images.append({
            'path': str(img_path),
            'filename': img_path.name,
            'category': category.replace('_', ' ').title(),
            'id': str(i) # String ID for pattern matching
        })

    return pd.DataFrame(images)

def encode_image_thumb(image_path):
    """Creates a small, optimised thumbnail for the grid."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail((300, 300)) # Small size for grid
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None

def encode_image_large(image_path):
    """Creates a larger version for the detail view."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail((600, 600)) # Larger size for detail
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None

# ============================================================================
# 🚀 APP INITIALISATION
# ============================================================================

df = load_data(CRISISMD_PATH)

# If data is missing, show a simple error layout
if df.empty:
    app = dash.Dash(__name__)
    app.layout = html.Div([html.H2("❌ Error: No images found. Check CRISISMD_PATH.")])
else:
    categories = sorted(df['category'].unique())
    
    # External CSS for nice fonts and scrollbars
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Crisis Data Explorer")

    # ============================================================================
    # 🖥️ VISUAL LAYOUT
    # ============================================================================
    app.layout = html.Div([
        
        # --- HEADER ---
        html.Div([
            html.H3("CrisisMMD Data Explorer", style={'margin': '0', 'fontWeight': '300', 'color': 'white'}),
            html.Div([
                html.Span("Dataset Status: ", style={'color': '#bdc3c7'}),
                html.Span("Ready", style={'color': '#2ecc71', 'fontWeight': 'bold'})
            ], style={'fontSize': '14px'})
        ], style={
            'padding': '15px 30px', 
            'backgroundColor': '#2c3e50', 
            'display': 'flex', 
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
        }),

        html.Div([
            # --- LEFT SIDEBAR (Filters & Details) ---
            html.Div([
                html.H5("Filter View", style={'marginTop': '0', 'color': '#34495e'}),
                html.Label("Event Type:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='category-filter',
                    options=[{'label': 'All Events', 'value': 'all'}] + [{'label': c, 'value': c} for c in categories],
                    value='all',
                    clearable=False
                ),
                
                html.Hr(style={'margin': '20px 0'}),
                
                # INSPECTOR PANEL
                html.Div(id='inspector-panel', children=[
                    html.Div([
                        html.H5("Inspector", style={'color': '#95a5a6', 'textAlign': 'center', 'marginTop': '50px'}),
                        html.P("Click an image in the grid to inspect details.", style={'textAlign': 'center', 'color': '#bdc3c7'})
                    ])
                ])

            ], style={
                'width': '300px', 
                'backgroundColor': '#f8f9fa', 
                'padding': '20px', 
                'borderRight': '1px solid #ddd',
                'height': 'calc(100vh - 70px)',
                'overflowY': 'auto',
                'boxSizing': 'border-box'
            }),

            # --- RIGHT MAIN AREA (The Grid) ---
            html.Div([
                html.H4(id='grid-header', children="All Images", style={'fontWeight': '300', 'marginBottom': '20px'}),
                
                # The Grid Container
                html.Div(id='image-grid', style={
                    'display': 'flex', 
                    'flexWrap': 'wrap', 
                    'gap': '15px',
                    'alignContent': 'flex-start'
                })

            ], style={
                'flex': '1', 
                'padding': '30px', 
                'height': 'calc(100vh - 70px)',
                'overflowY': 'auto',
                'backgroundColor': 'white'
            })

        ], style={'display': 'flex'}) # End of Flex Container

    ], style={'fontFamily': 'Helvetica, Arial, sans-serif', 'height': '100vh', 'overflow': 'hidden'})


    # ============================================================================
    # 🔌 INTERACTIVITY
    # ============================================================================

    # Callback 1: Update Grid based on Filter
    @app.callback(
        [Output('image-grid', 'children'),
         Output('grid-header', 'children')],
        [Input('category-filter', 'value')]
    )
    def update_grid(selected_category):
        if selected_category == 'all':
            filtered_df = df
            header_text = f"All Images ({len(df)})"
        else:
            filtered_df = df[df['category'] == selected_category]
            header_text = f"Category: {selected_category} ({len(filtered_df)})"

        grid_cards = []
        
        for i, row in filtered_df.iterrows():
            img_src = encode_image_thumb(row['path'])
            if img_src:
                # Each card is a Div with an image inside
                # We use 'n_clicks' to detect selection
                card = html.Div([
                    html.Img(src=img_src, style={'width': '100%', 'height': '150px', 'objectFit': 'cover', 'borderRadius': '4px 4px 0 0'}),
                    html.Div([
                        html.P(row['category'], style={'fontSize': '11px', 'color': '#7f8c8d', 'fontWeight': 'bold', 'margin': '0', 'textTransform': 'uppercase'}),
                        html.P(row['filename'][:15] + "...", style={'fontSize': '12px', 'color': '#2c3e50', 'margin': '5px 0 0 0'}),
                    ], style={'padding': '10px'})
                ], 
                # IMPORTANT: We assign an ID to this div so we know which one was clicked
                id={'type': 'grid-card', 'index': row['id']},
                n_clicks=0,
                style={
                    'width': '180px',
                    'backgroundColor': 'white',
                    'borderRadius': '5px',
                    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                    'cursor': 'pointer',
                    'transition': 'transform 0.1s',
                    'border': '1px solid #ecf0f1'
                })
                grid_cards.append(card)

        return grid_cards, header_text


    # Callback 2: Handle Card Click -> Update Inspector
    @app.callback(
        Output('inspector-panel', 'children'),
        [Input({'type': 'grid-card', 'index': ALL}, 'n_clicks')]
    )
    def display_details(n_clicks):
        # Find which card triggered the callback
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update

        # Get the ID of the clicked element (it returns a string like '{"index":"3","type":"grid-card"}.n_clicks')
        prop_id = ctx.triggered[0]['prop_id']
        
        # If nothing was actually clicked (initial load), stop
        if 'n_clicks' not in prop_id or all(c == 0 for c in n_clicks):
            return dash.no_update

        # Extract the ID dictionary from the string
        import json
        try:
            # Parse the JSON-like string ID manually or use eval (safe here as it's internal)
            # prop_id looks like: {"index":"0","type":"grid-card"}.n_clicks
            clicked_id_str = prop_id.split('.')[0]
            clicked_id_dict = json.loads(clicked_id_str)
            image_id = clicked_id_dict['index']
        except:
            return dash.no_update

        # Find the row in the dataframe
        row = df[df['id'] == image_id].iloc[0]
        large_img = encode_image_large(row['path'])

        # Return the Detail View
        return html.Div([
            html.H5("Image Details", style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            
            html.Div([
                html.Img(src=large_img, style={'width': '100%', 'borderRadius': '5px', 'marginBottom': '15px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})
            ], style={'textAlign': 'center'}),

            html.Label("Filename:", style={'fontWeight': 'bold', 'fontSize': '12px'}),
            html.P(row['filename'], style={'fontSize': '14px', 'fontFamily': 'monospace', 'backgroundColor': '#e1e5e8', 'padding': '5px', 'borderRadius': '3px'}),

            html.Label("Predicted Category:", style={'fontWeight': 'bold', 'fontSize': '12px', 'marginTop': '15px'}),
            html.Div(row['category'], style={
                'backgroundColor': '#d5f5e3', 
                'color': '#1e8449', 
                'padding': '8px', 
                'borderRadius': '4px', 
                'fontWeight': 'bold',
                'textAlign': 'center'
            }),

            html.Hr(),
            html.P("This is a verified data point from the CrisisMMD dataset.", style={'fontSize': '12px', 'color': '#95a5a6', 'fontStyle': 'italic'})
        ], style={'animation': 'fadeIn 0.3s'})

if __name__ == '__main__':
    app.run(debug=True, port=8050)