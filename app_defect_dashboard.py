"""Standalone entry point for the uploaded defect-analysis dashboard.

The existing ``app.py`` entry point remains unchanged. This entry point reuses
the current defect-metrics implementation, including CSV upload persistence,
filters, row selection, station plots, and chat context, without restoring the
visible 3-D viewer.
"""

from pathlib import Path


exec((Path(__file__).with_name("app_reactive_v2.py")).read_text(encoding="utf-8"), globals())