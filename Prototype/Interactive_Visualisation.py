import dash
from dash import dcc, html, Input, Output
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import random
import pandas as pd


CRISISMD_PATH = r"Prototype/data_image" 
MAX_IMAGES_TO_LOAD = 40  # Fewer images = cleaner screenshot



def load_data_simple(path_str):
    data_path = Path(path_str)
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files.extend(list(data_path.rglob(ext)))
    
    if not files: return pd.DataFrame()

    random.shuffle(files)
    selected = files[:MAX_IMAGES_TO_LOAD]
    
    images = []
    for i, p in enumerate(selected):
        try:
            cat = p.relative_to(data_path).parts[0] if len(p.relative_to(data_path).parts) > 1 else "will be natural disaster type"
        except:
            cat = "will be natural disaster type"
        
        images.append({
            'path': str(p),
            'filename': "this will be image filename",
            'category': cat.replace('_', ' ').title(),
            'id': i
        })
    return pd.DataFrame(images)

def get_thumbnail(path):
    try:
        with Image.open(path) as img:
            img.thumbnail((400, 400)) # High quality for screenshots
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
    except: return None



df = load_data_simple(CRISISMD_PATH)
categories = sorted(df['category'].unique()) if not df.empty else []

app = dash.Dash(__name__, title="Crisis App")

app.layout = html.Div([

    # --- TOP BAR (Fixed Header) ---
    html.Div([
        html.H2("prototype for 25/11/2025", style={'margin': '0', 'color': 'white', 'fontSize': '24px', 'fontWeight': '600'}),
        html.Div([
            dcc.Dropdown(
                id='filter',
                options=[{'label': 'Show All Events', 'value': 'all'}] + [{'label': c, 'value': c} for c in categories],
                value='all',
                clearable=False,
                style={'width': '200px', 'fontSize': '14px'}
            )
        ], style={'backgroundColor': 'white', 'borderRadius': '4px'})
    ], style={
        'position': 'fixed', 'top': '0', 'left': '0', 'right': '0',
        'height': '70px', 'backgroundColor': '#2c3e50',
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
        'padding': '0 20px', 'zIndex': '1000', 'boxShadow': '0 2px 10px rgba(0,0,0,0.2)'
    }),

    # --- MAIN GRID ---
    html.Div(id='grid-container', style={
        'marginTop': '90px', # Push down below fixed header
        'padding': '20px',
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'center',
        'gap': '20px'
    })

], style={'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif', 'backgroundColor': '#f4f6f7', 'minHeight': '100vh'})

# ============================================================================
# 🔌 LOGIC
# ============================================================================

@app.callback(
    Output('grid-container', 'children'),
    [Input('filter', 'value')]
)
def update_grid(cat):
    if df.empty: return html.P("No images found.")
    
    dff = df if cat == 'all' else df[df['category'] == cat]
    
    cards = []
    for _, row in dff.iterrows():
        src = get_thumbnail(row['path'])
        if src:
            # Simple Card Design
            card = html.Div([
                html.Img(src=src, style={
                    'width': '100%', 'height': '200px', 'objectFit': 'cover',
                    'borderTopLeftRadius': '8px', 'borderTopRightRadius': '8px'
                }),
                html.Div([
                    html.P(row['category'], style={
                        'margin': '0', 'fontSize': '12px', 'fontWeight': 'bold', 
                        'color': '#3498db', 'textTransform': 'uppercase'
                    }),
                    html.P(row['filename'], style={
                        'margin': '5px 0 0 0', 'fontSize': '11px', 'color': '#7f8c8d', 
                        'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'
                    })
                ], style={'padding': '12px'})
            ], style={
                'width': '220px', 'backgroundColor': 'white',
                'borderRadius': '8px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.05)',
                'overflow': 'hidden', 'transition': 'transform 0.2s'
            })
            cards.append(card)
            
    return cards

if __name__ == '__main__':
    app.run(debug=True, port=8050)