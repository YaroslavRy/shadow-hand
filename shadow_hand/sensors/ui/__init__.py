"""Native diagnostics UI modules."""

from .chrome import build_base_canvas, put_text
from .layout import DEFAULT_DIAGNOSTICS_LAYOUT, DiagnosticsLayout
from .palette import DEFAULT_PALETTE, DiagnosticsPalette

__all__ = [
    "build_base_canvas",
    "put_text",
    "DEFAULT_DIAGNOSTICS_LAYOUT",
    "DiagnosticsLayout",
    "DEFAULT_PALETTE",
    "DiagnosticsPalette",
]
