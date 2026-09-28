"""Streamlit entry point for the reactive lattice dashboard."""

from pathlib import Path


exec((Path(__file__).with_name("app_reactive_v2.py")).read_text(encoding="utf-8-sig"), globals())
