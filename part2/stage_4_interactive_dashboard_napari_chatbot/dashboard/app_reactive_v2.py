"""Streamlit client for the shared FastAPI/Napari defect-analysis broker."""

import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("DASHBOARD_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("DASHBOARD_API_TIMEOUT", "5"))
SUMMARY_TIMEOUT = float(os.getenv("DASHBOARD_SUMMARY_TIMEOUT", "30"))
st.set_page_config(page_title="Lattice NDE Dashboard", layout="wide")


def api_get(path: str, *, request_timeout: float = TIMEOUT, **params: Any) -> dict:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def load_struts() -> pd.DataFrame:
    payload = api_get("/dashboard/struts", request_timeout=SUMMARY_TIMEOUT)
    frame = pd.DataFrame(payload.get("items", []))
    if frame.empty or "strut_id" not in frame:
        raise ValueError("FastAPI returned no strut summary records")
    frame["strut_id"] = pd.to_numeric(frame["strut_id"], errors="raise").astype(int)
    return frame


def load_stations(strut_id: int) -> pd.DataFrame:
    return pd.DataFrame(api_get(f"/struts/{int(strut_id)}/stations").get("items", []))


def find_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {"".join(c for c in str(col).lower() if c.isalnum()): col for col in frame.columns}
    for alias in aliases:
        found = normalized.get("".join(c for c in alias.lower() if c.isalnum()))
        if found is not None:
            return found
    return None


def value(row: pd.Series, aliases: list[str], default: Any = "N/A") -> Any:
    column = find_column(pd.DataFrame([row]), aliases)
    result = row.get(column) if column else None
    return default if result is None or pd.isna(result) or result == "" else result


def text_series(frame: pd.DataFrame, column: str, default: str) -> pd.Series:
    return frame[column].fillna(default).astype(str) if column in frame else pd.Series(default, index=frame.index)


def add_severity(frame: pd.DataFrame) -> None:
    names = [name for name in ("missing_score", "broken_score", "thin_score", "inflated_score", "bend_score") if name in frame]
    if not names:
        frame["severity"] = "Unscored"
        return
    scores = frame[names].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0)
    nonzero = scores[scores > 0]
    q1, q2 = nonzero.quantile([0.33, 0.66]).tolist() if not nonzero.empty else (1, 2)
    frame["severity"] = np.select([scores <= q1, scores <= q2], ["Low", "Medium"], default="High")


def publish_selection(strut_id: int) -> None:
    strut_id = int(strut_id)
    if st.session_state.get("last_published_strut_id") == strut_id:
        return
    api_post("/select_struts", {"strut_ids": [strut_id]})
    st.session_state.last_published_strut_id = strut_id


def poll_selection() -> None:
    try:
        active_id = api_get("/state").get("active_strut_id")
    except requests.RequestException:
        return
    if active_id is not None and int(active_id) != st.session_state.get("selected_strut_id"):
        st.session_state.selected_strut_id = int(active_id)
        st.session_state.last_published_strut_id = int(active_id)
        st.rerun()


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def sync_selection():
        poll_selection()
else:
    def sync_selection():
        st.caption("Upgrade Streamlit to enable automatic broker polling.")

sync_selection()
st.title("Lattice Structure NDE Dashboard")
st.caption(f"FastAPI broker: {API_URL}")

try:
    summary = load_struts().copy()
except (requests.RequestException, ValueError) as exc:
    st.error(f"Could not load dashboard data from FastAPI at {API_URL}: {exc}")
    st.info("Start FastAPI with: uvicorn part2.stage_4_interactive_dashboard_napari_chatbot.run_fastapi:app --host 127.0.0.1 --port 8000")
    st.stop()

summary["primary_defect"] = text_series(summary, "primary_defect", "Unknown")
summary["stage2_classification"] = text_series(summary, "stage2_classification", "Unknown")
summary["needs_review"] = text_series(summary, "needs_review", "Not provided")
add_severity(summary)

st.sidebar.header("Inspection")
search = st.sidebar.text_input("Search strut_id", placeholder="e.g. 1234")
primary_values = sorted(summary["primary_defect"].unique())
stage2_values = sorted(summary["stage2_classification"].unique())
review_values = sorted(summary["needs_review"].unique())
selected_primary = st.sidebar.multiselect("Primary defect", primary_values, default=primary_values)
selected_stage2 = st.sidebar.multiselect("Stage 2 classification", stage2_values, default=stage2_values)
selected_review = st.sidebar.multiselect("Needs review", review_values, default=review_values)
unit_column = find_column(summary, ["unit_cell_ids", "unit_cell_id", "inventory_unit_cell_ids", "unit_cell"])
unit_values = sorted(summary[unit_column].dropna().astype(str).unique()) if unit_column else []
selected_units = st.sidebar.multiselect("Unit cell ID", unit_values, default=unit_values) if unit_values else []
severity_values = sorted(summary["severity"].unique())
selected_severity = st.sidebar.multiselect("Severity", severity_values, default=severity_values)

filtered = summary[
    summary["primary_defect"].isin(selected_primary)
    & summary["stage2_classification"].isin(selected_stage2)
    & summary["needs_review"].isin(selected_review)
    & summary["severity"].isin(selected_severity)
].copy()
if unit_column and selected_units:
    filtered = filtered[filtered[unit_column].astype(str).isin(selected_units)]
if search.strip():
    filtered = filtered[filtered["strut_id"].astype(str).str.contains(search.strip(), regex=False)]
if filtered.empty:
    st.warning("No struts match the current filters.")
    st.stop()
filtered = filtered.sort_values("strut_id").reset_index(drop=True)

st.subheader("Defect summary")
st.caption("Select a row to publish that strut as the shared active selection.")
table_event = st.dataframe(filtered.drop(columns=["severity"], errors="ignore"), width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", key="broker_strut_summary")
selected_rows = getattr(getattr(table_event, "selection", None), "rows", [])
if selected_rows:
    clicked_id = int(filtered.iloc[selected_rows[0]]["strut_id"])
    st.session_state.selected_strut_id = clicked_id
    try:
        publish_selection(clicked_id)
    except requests.RequestException as exc:
        st.error(f"Could not publish selection to FastAPI: {exc}")

all_ids = summary["strut_id"].astype(int).tolist()
current = int(st.session_state.get("selected_strut_id", filtered.iloc[0]["strut_id"]))
if current not in all_ids:
    current = int(filtered.iloc[0]["strut_id"])
options = filtered["strut_id"].astype(int).tolist()
if current not in options:
    options = [current] + options
selected_id = st.sidebar.selectbox("Active strut", options, index=options.index(current))
if int(selected_id) != st.session_state.get("selected_strut_id"):
    st.session_state.selected_strut_id = int(selected_id)
    try:
        publish_selection(int(selected_id))
    except requests.RequestException as exc:
        st.error(f"Could not publish selection to FastAPI: {exc}")
selected_id = int(st.session_state.selected_strut_id)
selected_summary = summary[summary["strut_id"] == selected_id].iloc[0]

st.subheader(f"Selected strut: {selected_id}")
st.markdown(f"**Defect badge:** `{value(selected_summary, ['primary_defect'])}`")
left, middle, right = st.columns(3)
with left:
    st.write({label: value(selected_summary, aliases) for label, aliases in (
        ("primary_defect", ["primary_defect"]), ("secondary_defects", ["secondary_defects", "secondary_defect"]),
        ("stage2_classification", ["stage2_classification", "classification"]), ("confidence", ["confidence", "confidence_score"]),
        ("needs_review", ["needs_review", "review_required"]),)})
with middle:
    st.write({name: value(selected_summary, [name]) for name in ("missing_score", "broken_score", "thin_score", "inflated_score", "bend_score")})
with right:
    st.write({label: value(selected_summary, aliases) for label, aliases in (
        ("sampled_occupancy", ["sampled_occupancy", "occupancy"]), ("sampled_mean_intensity", ["sampled_mean_intensity", "mean_intensity"]),
        ("cross-section radius", ["median_cross_section_radius_um", "equivalent_radius_um"]), ("max_centerline_offset_um", ["max_centerline_offset_um", "centerline_offset_um"]),
        ("rms_centerline_offset_um", ["rms_centerline_offset_um", "centerline_offset_um"]), ("bend_curvature_um", ["bend_curvature_um", "curvature_um"]),)})

st.subheader("Station-level defect analysis")
try:
    stations = load_stations(selected_id)
except requests.RequestException as exc:
    stations = pd.DataFrame()
    st.error(f"Could not load station data for strut {selected_id}: {exc}")
position_column = find_column(stations, ["position_fraction", "station_fraction", "position"])
if stations.empty or position_column is None:
    st.warning("No station rows with position_fraction were found for the active strut.")
else:
    stations["__position_fraction"] = pd.to_numeric(stations[position_column], errors="coerce")
    stations = stations.dropna(subset=["__position_fraction"]).sort_values("__position_fraction")
    x = stations["__position_fraction"]
    charts = st.columns(2)

    def series(aliases: list[str]):
        column = find_column(stations, aliases)
        return column, pd.to_numeric(stations[column], errors="coerce") if column else None

    offset_col, offsets = series(["centroid_offset_um", "centroid_offset", "centerline_offset_um"])
    occupancy_col, occupancies = series(["material_occupancy", "occupancy", "material_fraction"])
    radius_col, radii = series(["equivalent_radius_um", "equivalent_radius", "radius_um"])
    intensity_col, intensities = series(["mean_intensity", "sampled_mean_intensity"])
    with charts[0]:
        if offset_col:
            fig = go.Figure(go.Scatter(x=x, y=offsets, mode="lines+markers", name=offset_col))
            valid = offsets.dropna()
            if not valid.empty:
                index = valid.idxmax()
                fig.add_trace(go.Scatter(x=[x.loc[index]], y=[offsets.loc[index]], mode="markers+text", text=["maximum offset"], textposition="top center", marker={"color": "red", "size": 11}))
            fig.update_layout(title="Centroid offset vs position_fraction", xaxis_title="position_fraction", yaxis_title=offset_col)
            st.plotly_chart(fig, width="stretch")
        if occupancy_col:
            fig = go.Figure(go.Scatter(x=x, y=occupancies, mode="lines+markers", name=occupancy_col))
            low = occupancies < 0.5
            if low.any():
                fig.add_trace(go.Scatter(x=x[low], y=occupancies[low], mode="markers", marker={"color": "red", "size": 9}, name="Low occupancy"))
            fig.update_layout(title="Material occupancy vs position_fraction", xaxis_title="position_fraction", yaxis_title=occupancy_col)
            st.plotly_chart(fig, width="stretch")
    with charts[1]:
        if radius_col:
            fig = go.Figure(go.Scatter(x=x, y=radii, mode="lines+markers", name=radius_col))
            fig.update_layout(title="Equivalent radius vs position_fraction", xaxis_title="position_fraction", yaxis_title=radius_col)
            st.plotly_chart(fig, width="stretch")
        if intensity_col:
            fig = go.Figure(go.Scatter(x=x, y=intensities, mode="lines+markers", name=intensity_col))
            fig.update_layout(title="Mean intensity vs position_fraction", xaxis_title="position_fraction", yaxis_title=intensity_col)
            st.plotly_chart(fig, width="stretch")

with st.expander("Selected station rows"):
    st.dataframe(stations.drop(columns=["__position_fraction"], errors="ignore"), width="stretch", hide_index=True)

st.subheader("Chat")
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
question = st.chat_input("Ask about the active strut or defect summary")
if question:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    try:
        response = api_post("/chat", {"message": question, "active_strut_id": selected_id})
        st.session_state.chat_messages.append({"role": "assistant", "content": response.get("reply", "FastAPI returned no reply.")})
        st.rerun()
    except requests.RequestException as exc:
        st.session_state.chat_messages.append({"role": "assistant", "content": f"FastAPI chat is unavailable: {exc}"})
        st.rerun()
