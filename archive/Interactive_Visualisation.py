"""
Interactive Image Gallery for CrisisMMD Dataset.

A Dash web application for displaying disaster images in a grid layout
filtered by event category.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""

import dash
from dash import dcc, html, Input, Output
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import random
import pandas as pd


# Configuration
DATASET_PATH = r"Prototype/data_image"
MAX_IMAGES_TO_LOAD = 40


def load_data(path_str):
    """
    Load image metadata from the dataset folder.
    
    This function scans the given directory for image files, randomly
    selects a subset, and returns their paths and categories as a DataFrame.
    
    Args:
        path_str: Path to the image directory.
        
    Returns:
        DataFrame containing image paths and categories.
    """
    data_path = Path(path_str)
    files = []
    
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files.extend(list(data_path.rglob(ext)))
    
    if not files:
        return pd.DataFrame()

    random.shuffle(files)
    selected = files[:MAX_IMAGES_TO_LOAD]
    
    images = []
    for i, filepath in enumerate(selected):
        # Extract category from folder name
        try:
            relative = filepath.relative_to(data_path)
            if len(relative.parts) > 1:
                category = relative.parts[0]
            else:
                category = "uncategorised"
        except Exception:
            category = "uncategorised"
        
        images.append({
            "path": str(filepath),
            "filename": filepath.name,
            "category": category.replace("_", " ").title(),
            "id": i
        })
    
    return pd.DataFrame(images)


def create_thumbnail(path):
    """
    Create a base64-encoded thumbnail from an image file.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Base64 data URL string, or None if the image cannot be loaded.
    """
    try:
        with Image.open(path) as img:
            img.thumbnail((400, 400))
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return "data:image/jpeg;base64,{}".format(encoded)
    except Exception:
        return None


# Load the data
df = load_data(DATASET_PATH)
categories = sorted(df["category"].unique()) if not df.empty else []

# Create the Dash app
app = dash.Dash(__name__, title="Crisis Image Viewer")

# Define the layout
app.layout = html.Div([

    # Header bar
    html.Div([
        html.H2(
            "Disaster Image Viewer", 
            style={
                "margin": "0", 
                "color": "white", 
                "fontSize": "24px", 
                "fontWeight": "600"
            }
        ),
        html.Div([
            dcc.Dropdown(
                id="category-filter",
                options=[{"label": "Show All Events", "value": "all"}] + 
                        [{"label": c, "value": c} for c in categories],
                value="all",
                clearable=False,
                style={"width": "200px", "fontSize": "14px"}
            )
        ], style={"backgroundColor": "white", "borderRadius": "4px"})
    ], style={
        "position": "fixed",
        "top": "0",
        "left": "0",
        "right": "0",
        "height": "70px",
        "backgroundColor": "#2c3e50",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "padding": "0 20px",
        "zIndex": "1000",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.2)"
    }),

    # Image grid
    html.Div(id="image-grid", style={
        "marginTop": "90px",
        "padding": "20px",
        "display": "flex",
        "flexWrap": "wrap",
        "justifyContent": "center",
        "gap": "20px"
    })

], style={
    "fontFamily": "Segoe UI, Roboto, Helvetica, Arial, sans-serif",
    "backgroundColor": "#f4f6f7",
    "minHeight": "100vh"
})


@app.callback(
    Output("image-grid", "children"),
    [Input("category-filter", "value")]
)
def update_grid(selected_category):
    """
    Update the image grid based on the selected category filter.
    
    Args:
        selected_category: The category to filter by, or 'all' for no filter.
        
    Returns:
        List of card components for each image.
    """
    if df.empty:
        return html.P("No images found.")
    
    # Filter the dataframe
    if selected_category == "all":
        filtered = df
    else:
        filtered = df[df["category"] == selected_category]
    
    cards = []
    for _, row in filtered.iterrows():
        thumbnail = create_thumbnail(row["path"])
        
        if thumbnail:
            card = html.Div([
                html.Img(src=thumbnail, style={
                    "width": "100%",
                    "height": "200px",
                    "objectFit": "cover",
                    "borderTopLeftRadius": "8px",
                    "borderTopRightRadius": "8px"
                }),
                html.Div([
                    html.P(row["category"], style={
                        "margin": "0",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                        "color": "#3498db",
                        "textTransform": "uppercase"
                    }),
                    html.P(row["filename"], style={
                        "margin": "5px 0 0 0",
                        "fontSize": "11px",
                        "color": "#7f8c8d",
                        "overflow": "hidden",
                        "textOverflow": "ellipsis",
                        "whiteSpace": "nowrap"
                    })
                ], style={"padding": "12px"})
            ], style={
                "width": "220px",
                "backgroundColor": "white",
                "borderRadius": "8px",
                "boxShadow": "0 4px 6px rgba(0,0,0,0.05)",
                "overflow": "hidden",
                "transition": "transform 0.2s"
            })
            cards.append(card)
    
    return cards


if __name__ == "__main__":
    app.run(debug=True, port=8050)