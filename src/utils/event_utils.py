"""
Shared Event Utilities for CrisisMMD Dataset.

This module provides event parsing and mapping functions used across
multiple modules (backend, evaluation, etc.) to avoid duplication.

Author: Rashid
Supervisor: XinHui Ma
Project: Visualising Natural Disaster Image Embeddings
"""


# Event Mappings based on CrisisMMD folder structure
EVENT_MAPPINGS = {
    "california_wildfires": "California Wildfires",
    "hurricane_harvey": "Hurricane Harvey",
    "hurricane_irma": "Hurricane Irma",
    "hurricane_maria": "Hurricane Maria",
    "iraq_iran_earthquake": "Iraq-Iran Earthquake",
    "mexico_earthquake": "Mexico Earthquake",
    "srilanka_floods": "Sri Lanka Floods",
}


def parse_event(path: str) -> str:
    """
    Extract the event type from an image file path.

    The CrisisMMD dataset has folder names like 'california_wildfires',
    'hurricane_harvey', etc. This function parses those labels.

    Args:
        path: Full path to the image file.

    Returns:
        Human-readable event name (e.g. 'California Wildfires').
    """
    path = str(path).replace("\\", "/").lower()

    for key, label in EVENT_MAPPINGS.items():
        if key in path:
            return label

    return "Unknown Event"
