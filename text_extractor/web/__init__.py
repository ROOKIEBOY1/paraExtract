"""Persistent extraction services and local HTTP integration."""

from .application import (
    ExtractionService,
    SwitchingExtractionService,
    WebApplication,
    load_text_samples,
    load_web_samples,
)
from .server import make_handler

__all__ = [
    "ExtractionService",
    "SwitchingExtractionService",
    "WebApplication",
    "load_text_samples",
    "load_web_samples",
    "make_handler",
]
