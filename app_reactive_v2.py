"""Streamlit defect-inspection dashboard."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google import genai

ROOT = Path(__file__).parent
UPLOADS = ROOT / "outputs/streamlit_uploads"
st.set_page_config(page_title="Lattice NDE Dashboard", layout="wide")

SUMMARY = {
    "primary_defect": ["primary_defect", "primary defect", "defect", "defect_type", "defect type"],
    "stage2_classification": ["stage2_classification", "stage2 classification", "stage2a_classification", "classification"],
    "needs_review": ["needs_review", "needs review", "review_required", "review required"],
    "severity": ["severity", "severity_level", "severity level"],
    "diameter": ["diameter", "equivalent_diameter", "effective_diameter_um", "median_effective_diameter_um", "diameter_um"],
    "nominal": ["nominal_diameter", "nominal diameter", "nominal_diameter_um", "design_diameter", "cad_diameter"],
    "ratio": ["diameter_ratio", "diameter_to_nominal", "diameter_to_nominal_ratio", "diameter / nominal"],
    "deviation": ["max_centerline_deviation", "max_centerline_deviation_um", "centerline_deviation", "maximum_centerline_offset_um"],
    "occupancy": ["occupancy", "material_occupancy", "material fraction", "ct_material_occupancy", "station_material_fraction"],
    "curvature": ["curvature", "tortuosity", "bend_curvature"],
}
STATION = {
    "position": ["position_fraction", "station_fraction", "position", "fraction"],
    "occupancy": ["material_occupancy", "occupancy", "material_fraction", "station_material_fraction"],
    "diameter": ["equivalent_diameter", "diameter", "effective_diameter_um", "diameter_um", "equivalent_radius", "radius", "radius_um"],
    "offset": ["centroid_offset", "centroid_offset_um", "centerline_offset_um", "centerline_deviation", "centerline_deviation_um"],
}

def key(x): return "".join(c for c in str(x).lower() if c.isalnum())
def col(df, aliases):
    lookup = {key(c): c for c in df.columns}
    return next((lookup[key(a)] for a in aliases if key(a) in lookup), None)
def load_frame(path, label):
    df = pd.read_csv(path)
    c = col(df, ["strut_id", "strut id", "strut", "id"])
    if c is None: raise ValueError(f"{label} needs a strut_id column.")
    df["strut_id"] = pd.to_numeric(df[c], errors="coerce")
    if df.strut_id.isna().any(): raise ValueError(f"{label} contains invalid strut IDs.")
    df["strut_id"] = df.strut_id.astype(int)
    return df
def canon(df, aliases):
    df = df.copy()
    for name, options in aliases.items():
        c = col(df, options)
        if c and name not in df: df[name] = df[c]
    return df
def truthy(x): return str(x).strip().lower() in {"true", "1", "yes", "required", "needs review", "needs_review"}
def gemini_chat_response(question, selected, stations, chat_history):
    """Generate a response grounded in the active strut and current chat."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini chat is unavailable because GEMINI_API_KEY is not set."

    selected_summary = json.dumps(selected.to_dict(), default=str, indent=2)
    station_records = stations.drop(columns=["__position_pct"], errors="ignore").to_dict(orient="records")
    station_data = json.dumps(station_records, default=str, indent=2)
    transcript = json.dumps(chat_history, default=str, indent=2)
    prompt = f"""You are a defect-inspection assistant for a lattice NDE dashboard.
Answer the user's question using only the inspection context below. Be concise,
clearly distinguish measured values from interpretation, and say when the data
does not support a conclusion. Do not invent missing values.

Selected strut summary:
{selected_summary}

Station-level data for the selected strut:
{station_data}

Conversation history:
{transcript}

User's latest question:
{question}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
    )
    return response.text or "Gemini returned an empty response."
def display(x, unit="", digits=2):
    if x is None or pd.isna(x) or x == "": return "N/A"
    n = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return f"{float(n):,.{digits}f} {unit}".strip() if pd.notna(n) else str(x)
def chart(fig, height=300):
    fig.update_layout(height=height, margin=dict(l=10,r=10,t=42,b=10), paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#e5e7eb", xaxis_gridcolor="#293750", yaxis_gridcolor="#293750")
    return fig
def metric(label, value, help_text=""):
    title = f' title="{help_text}"' if help_text else ""
    st.markdown(f'<div class="metric-card"{title}><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
def series(df, aliases):    c = col(df, aliases); return c, pd.to_numeric(df[c], errors="coerce") if c else (None, None)

def save_upload(upload, state_key):
    if upload is None: return
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / (Path(upload.name).name.rstrip(" .") or f"{state_key}.csv")
    if not path.exists() or path.stat().st_size != upload.size: path.write_bytes(upload.getbuffer())
    st.session_state[state_key] = str(path)

@st.cache_data
def read_csv(path): return pd.read_csv(path)
def data():
    a, b = st.session_state.get("summary_path"), st.session_state.get("station_path")
    if not a or not b or not Path(a).exists() or not Path(b).exists(): return None, None, None
    try:
        summary = canon(load_frame(a, "Strut summary CSV"), SUMMARY)
        station = canon(load_frame(b, "Station measurements CSV"), STATION)
        if summary.strut_id.duplicated().any(): raise ValueError("The strut summary must contain one row per strut ID.")
        return summary, station, None
    except Exception as e: return None, None, str(e)
def severity(df):
    if "severity" in df: return df
    scores = [c for c in ["missing_score","broken_score","thin_score","inflated_score","bend_score"] if c in df]
    if not scores: return df
    values = df[scores].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0); nz = values[values > 0]
    q1, q2 = nz.quantile([.33,.66]).tolist() if len(nz) else (1,2)
    result = df.copy(); result["severity"] = np.select([values <= q1, values <= q2], ["Low","Medium"], default="High"); return result

def overview(summary, station):
    st.title("Overview"); st.caption("High-level view of the defect-inspection workspace.")
    with st.container(border=True):
        st.subheader("Inspection workspace")
        st.write("Use Strut Analysis to upload inspection data, filter the review queue, and inspect station-level evidence.")
        if summary is None or station is None:
            st.info("No analysis data loaded yet. Open Strut Analysis to add the summary and station CSV files.")
        else:
            st.success(f"Analysis data ready  {len(summary):,} struts and {len(station):,} station records available.")
            st.write("Inspection Workflow documents the pipeline from source data through dashboard review.")
def flowchart():
    st.title("Inspection Workflow"); st.caption("The inspection pipeline from source data to human review.")
    steps = [("CAD + CT scan","CAD provides nominal geometry; CT provides the measured lattice volume."),("Registration","Aligns CAD and CT coordinates."),("Missing-strut analysis","Finds absent or unexpected material relative to design."),("Station-level morphology analysis","Measures occupancy, diameter/radius, and centerline offset along each strut."),("Defect classification","Combines measurements into defect labels and stage 2 classifications."),("Dashboard review","Supports filtering, strut inspection, local decisions, and contextual questions.")]
    for i,(title,desc) in enumerate(steps):
        a, arrow, b = st.columns([1,.12,2.5]); a.markdown(f"<div class='flow-stage'><div class='flow-number'>{i+1:02d}</div><div class='flow-title'>{title}</div></div>", unsafe_allow_html=True)
        if i < len(steps)-1: arrow.markdown("<div class='flow-arrow'></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='flow-description'>{desc}</div>", unsafe_allow_html=True)

def analysis(summary, station):
    st.title("Strut Analysis"); st.caption("Upload results, review matching struts, and inspect station-level evidence.")
    with st.container(border=True):
        st.subheader("Load analysis data")
        left, right = st.columns(2)
        with left: up1 = st.file_uploader("Strut summary CSV", type=["csv"], key="summary_upload")
        with right: up2 = st.file_uploader("Station measurements CSV", type=["csv"], key="station_upload")
        if up1: save_upload(up1, "summary_path")
        if up2: save_upload(up2, "station_path")
        st.button("Load analysis data", type="primary", use_container_width=True)
        a, b = st.session_state.get("summary_path"), st.session_state.get("station_path")
        if a and b and Path(a).exists() and Path(b).exists(): st.success(f"Analysis data ready - {Path(a).name} + {Path(b).name}")
        elif a or b: st.info("One file is saved. Select the other CSV to complete the dataset.")
        else: st.info("No analysis data loaded yet.")
    if summary is None or station is None:
        st.info("Load both CSVs above to use filters, selected-strut details, charts, and chat."); return

    summary = severity(summary.copy())
    for name, default in [("primary_defect", "Unknown"), ("stage2_classification", "Unknown"), ("needs_review", "Not provided"), ("severity", "Unscored")]:
        if name not in summary: summary[name] = default
    filter_keys = ["filter_search", "filter_primary", "filter_stage2", "filter_review", "filter_severity"]
    source_columns = {
        "occupancy": col(summary, SUMMARY["occupancy"]),
        "deviation": col(summary, SUMMARY["deviation"]),
        "ratio": col(summary, SUMMARY["ratio"] + ["diameter_ratio_to_nominal"]),
        "diameter": col(summary, SUMMARY["diameter"]),
        "bent": col(summary, ["curvature", "tortuosity", "tortuosity_ratio", "bend_curvature"]),
        "review": col(summary, SUMMARY["needs_review"]),
    }
    ratio_or_diameter = source_columns["ratio"] or source_columns["diameter"]
    quick_requirements = {
        "Lowest occupancy": source_columns["occupancy"],
        "Largest deviation": source_columns["deviation"],
        "Thinnest": ratio_or_diameter,
        "Thickest/inflated": ratio_or_diameter,
        "Most bent": source_columns["bent"],
        "Needs review": source_columns["review"],
    }
    quick_actions = [("Lowest occupancy", "lowest_occupancy"), ("Largest deviation", "largest_deviation"), ("Thinnest", "thinnest"), ("Thickest/inflated", "thickest_inflated"), ("Most bent", "most_bent"), ("Needs review", "needs_review")]

    def quick_candidates(choice):
        frame = summary.copy()
        if choice == "Needs review":
            frame = frame[frame.needs_review.map(truthy)]
            return frame.sort_values("strut_id"), int(frame.iloc[0].strut_id) if len(frame) else None
        aliases = {
            "Lowest occupancy": SUMMARY["occupancy"],
            "Largest deviation": SUMMARY["deviation"],
            "Thinnest": SUMMARY["ratio"] + ["diameter_ratio_to_nominal"] if source_columns["ratio"] else SUMMARY["diameter"],
            "Thickest/inflated": SUMMARY["ratio"] + ["diameter_ratio_to_nominal"] if source_columns["ratio"] else SUMMARY["diameter"],
            "Most bent": ["curvature", "tortuosity", "tortuosity_ratio", "bend_curvature"],
        }
        source = col(frame, aliases[choice])
        if source is None: return frame.sort_values("strut_id"), None
        ranked = frame.copy(); ranked["__quick_metric"] = pd.to_numeric(ranked[source], errors="coerce")
        ascending = choice in {"Lowest occupancy", "Thinnest"}
        ranked = ranked.dropna(subset=["__quick_metric"]).sort_values(["__quick_metric", "strut_id"], ascending=[ascending, True]).drop(columns="__quick_metric")
        return ranked, int(ranked.iloc[0].strut_id) if len(ranked) else None

    def activate_quick_pick(choice):
        candidates, picked = quick_candidates(choice)
        if picked is None: return
        for filter_key in filter_keys: st.session_state.pop(filter_key, None)
        st.session_state.quick_pick_state = choice
        st.session_state.show_matching_struts = True
        st.session_state.selected_strut_id = picked
        st.session_state.selected_strut_picker = picked

    def reset_filters():
        for filter_key in filter_keys: st.session_state.pop(filter_key, None)
        st.session_state.pop("quick_pick_state", None)
        st.session_state.pop("selected_strut_id", None)
        st.session_state.pop("selected_strut_picker", None)
        st.session_state.show_matching_struts = False

    with st.container(border=True):
        st.subheader("Inspection Start")
        search = st.text_input("Strut ID", placeholder="Search strut ID, e.g. 1728", key="filter_search")
        primary, stage2, review, sev = [], [], [], []
        st.caption(f"Loaded dataset: {len(summary):,} struts")
        quick_picks = st.columns(6)
        for button, (label, action) in zip(quick_picks, quick_actions):
            available = quick_requirements[label] is not None
            help_text = None if available else f"Unavailable: required metric for {label.lower()} is not present in the uploaded data."
            button.button(label, key=f"quick_pick_{action}", disabled=not available, help=help_text, use_container_width=True, on_click=activate_quick_pick, args=(label,))
        st.button("Reset filters", key="reset_filters", use_container_width=False, on_click=reset_filters)

    quick_state = st.session_state.get("quick_pick_state")
    quick_active = bool(quick_state and not search.strip() and not primary and not stage2 and not review and not sev)
    if quick_active:
        filtered, quick_selected = quick_candidates(quick_state)
        st.info(f"Quick pick active: {quick_state} - showing top candidates.")
    else:
        mask = pd.Series(True, index=summary.index)
        if primary: mask &= summary.primary_defect.astype(str).isin(primary)
        if stage2: mask &= summary.stage2_classification.astype(str).isin(stage2)
        if review: mask &= summary.needs_review.astype(str).isin(review)
        if sev: mask &= summary.severity.astype(str).isin(sev)
        filtered = summary[mask].copy()
        if search.strip(): filtered = filtered[filtered.strut_id.astype(str).str.contains(search.strip(), regex=False)]
        filtered = filtered.sort_values("strut_id")
    ids = filtered.strut_id.astype(int).tolist()
    st.caption(f"Showing {len(ids):,} of {len(summary):,} struts.")
    browse = st.button("Browse matching struts", key="browse_matching_struts", use_container_width=False)
    if browse: st.session_state.show_matching_struts = not st.session_state.get("show_matching_struts", False)
    table_event = None
    if st.session_state.get("show_matching_struts", False):
        useful = [x for x in ["strut_id", "primary_defect", "stage2_classification", "severity", "needs_review"] if x in filtered]
        table_event = st.dataframe(filtered[useful], hide_index=True, height=420, use_container_width=True, on_select="rerun", selection_mode="single-row", key="defect_queue")
    if not ids:
        st.warning("No uploaded struts match the current filters."); return

    previous = st.session_state.get("selected_strut_id")
    current = int(previous) if previous in ids else ids[0]
    rows = getattr(getattr(table_event, "selection", None), "rows", [])
    if rows:
        current = int(filtered.iloc[rows[0]].strut_id)
        st.session_state.selected_strut_id = current
    else:
        st.session_state.selected_strut_id = current
    st.caption(f"Selected strut: {current}. Browse the full result set above and select a row to inspect it.")
    selected = summary[summary.strut_id == int(current)].iloc[0]
    stations = station[station.strut_id == int(current)].copy()
    _, position = series(stations, STATION["position"])
    if position is not None:
        position = pd.to_numeric(position, errors="coerce")
        position = position * 100 if position.dropna().max() <= 1.0 else position
        stations["__position_pct"] = position
        stations = stations.dropna(subset=["__position_pct"]).sort_values("__position_pct")
        position = stations["__position_pct"]

    def value(aliases):
        source = col(summary, aliases); return selected[source] if source else None
    def metric_number(aliases, source_frame=summary, source_row=selected):
        source = col(source_frame, aliases)
        if not source: return None, None
        return pd.to_numeric(pd.Series([source_row[source]]), errors="coerce").iloc[0], source
    def scale_to_um(number, source_name):
        if number is None or pd.isna(number): return number
        name = str(source_name).lower()
        if "_mm" in name or "(mm" in name: return float(number) * 1000
        if "_m" in name or "(m" in name or name.endswith("meter"): return float(number) * 1000000
        return float(number)
    def fmt(number, digits=2, unit=""):
        if number is None or pd.isna(number): return "N/A"
        return f"{float(number):,.{digits}f} {unit}".strip()
    def pct(number, digits=1):
        if number is None or pd.isna(number): return "N/A"
        number = float(number) * 100 if abs(float(number)) <= 1 else float(number)
        return f"{number:.{digits}f}%"

    st.divider(); st.subheader(f"Selected strut {current}")
    measured, measured_source = metric_number(SUMMARY["diameter"])
    measured = scale_to_um(measured, measured_source)
    nominal, nominal_source = metric_number(SUMMARY["nominal"])
    nominal = scale_to_um(nominal, nominal_source)
    ratio, ratio_source = metric_number(SUMMARY["ratio"])
    if ratio is None and measured is not None and nominal not in (None, 0): ratio = measured / nominal
    deviation, deviation_source = metric_number(SUMMARY["deviation"])
    deviation = scale_to_um(deviation, deviation_source)
    rms, rms_source = metric_number(["rms_offset", "rms_centerline_offset", "rms_centerline_deviation", "rms_offset_um"])
    rms = scale_to_um(rms, rms_source)
    tortuosity, tortuosity_source = metric_number(["tortuosity", "tortuosity_ratio", "curvature", "bend_curvature"])
    occupancy, occupancy_source = metric_number(SUMMARY["occupancy"])
    cards = st.columns(5 if nominal is not None else 4)
    with cards[0]: metric("Occupancy", pct(occupancy), "Material occupancy along stations.")
    with cards[1]: metric("Measured diameter", fmt(measured, 1, "µm"), "Measured strut diameter.")
    if nominal is not None:
        with cards[2]: metric("Diameter / nominal", fmt(ratio, 3, "ratio"), f"Measured {fmt(measured, 1, 'µm')} vs nominal {fmt(nominal, 1, 'µm')}.")
        metric_offset = 3
    else: metric_offset = 2
    with cards[metric_offset]: metric("Maximum deviation", fmt(deviation, 1, "µm"), "Largest centerline deviation from the nominal path.")
    if rms is not None:
        with cards[metric_offset + 1]: metric("RMS offset", fmt(rms, 1, "µm"), "Root-mean-square centerline offset.")
    else:
        with cards[metric_offset + 1]: metric("Tortuosity ratio", fmt(tortuosity, 3, "unitless"), "Path length divided by straight-line length.")

    decisions = st.session_state.setdefault("review_choices", {}); buttons = st.columns(3)
    for button, label, decision in zip(buttons, ["Confirm defect", "Mark needs review", "Mark nominal"], ["confirmed_defect", "needs_review", "nominal"]):
        if button.button(label, key=f"review_{decision}_{current}", use_container_width=True): decisions[int(current)] = decision; st.rerun()
    if decisions.get(int(current)): st.caption(f"Local review decision: {decisions[int(current)].replace('_', ' ')}")

    def station_values(aliases, scale=False):
        source, values = series(stations, aliases)
        if source is None or values is None: return None, None
        values = pd.to_numeric(values, errors="coerce")
        if scale: values = values.map(lambda number: scale_to_um(number, source))
        return source, values
    def make_trend(title, y_title, aliases, scale=False, percent=False, minimum=False, maximum=False, reference=None):
        source, values = station_values(aliases, scale)
        if source is None or position is None or values.dropna().empty: return False
        plot_values = values * 100 if percent and values.dropna().max() <= 1 else values
        valid = pd.DataFrame({"position": position, "value": plot_values}).dropna()
        if valid.empty: return False
        fig = go.Figure(go.Scatter(x=valid.position, y=valid.value, mode="lines+markers", line={"color":"#38bdf8"}, name=y_title))
        if reference is not None and not pd.isna(reference): fig.add_hline(y=reference, line_dash="dash", line_color="#f59e0b", annotation_text="Nominal")
        point = valid.loc[valid.value.idxmin() if minimum else valid.value.idxmax()]
        label = "Lowest" if minimum else "Maximum"
        fig.add_trace(go.Scatter(x=[point.position], y=[point.value], mode="markers+text", text=[label], textposition="top center", marker={"color":"#f97316", "size":10}, name=label))
        fig.update_layout(title=title, xaxis_title="Position along strut (%)", yaxis_title=y_title)
        st.plotly_chart(chart(fig, 300), use_container_width=True)
        return float(point.value), float(point.position), label

    st.markdown("**Inspection summary**")
    summary_items = []
    main_specs = [("occupancy", "Material occupancy (%)", STATION["occupancy"], False, True, True, False, None), ("diameter", "Measured diameter (µm)", STATION["diameter"], True, False, True, False, nominal), ("offset", "Centerline offset (µm)", STATION["offset"], True, False, False, True, None)]
    for key_name, title, aliases, scale, percent, minimum, maximum, reference in main_specs:
        source, values = station_values(aliases, scale)
        if source is None or position is None or values.dropna().empty: continue
        plotted = values * 100 if percent and values.dropna().max() <= 1 else values
        valid = pd.DataFrame({"position": position, "value": plotted}).dropna()
        if valid.empty: continue
        point = valid.loc[valid.value.idxmin() if minimum else valid.value.idxmax()]
        name = "Lowest occupancy" if key_name == "occupancy" else "Minimum diameter" if key_name == "diameter" else "Maximum deviation"
        unit = "%" if percent else "µm"
        summary_items.append(f"{name}: {point.value:.1f}{unit} at {point.position:.0f}% of strut length")
    if summary_items: st.caption(" | ".join(summary_items))

    st.markdown("**Station trends**")
    trend_specs = [("Material occupancy", "Material occupancy (%)", STATION["occupancy"], False, True, True, False, None), ("Measured diameter", "Measured diameter (µm)", STATION["diameter"], True, False, True, False, nominal), ("Centerline offset", "Centerline offset (µm)", STATION["offset"], True, False, False, True, None)]
    rendered = []
    for spec in trend_specs:
        source, values = station_values(spec[2], spec[3])
        if source is not None and position is not None and not values.dropna().empty: rendered.append(spec)
    for panel, spec in zip(st.columns(max(1, len(rendered))), rendered):
        with panel: make_trend(*spec)

    optional = [("Mean CT intensity", "Mean CT intensity (a.u.)", ["mean_intensity", "intensity", "ct_intensity"], False, False, False, True, None), ("Outside-nominal material", "Outside-nominal material (%)", ["outside_nominal_material", "outside_nominal_fraction", "outside_nominal", "excess_material_fraction"], False, True, True, False, None)]
    optional_available = []
    for spec in optional:
        source, values = station_values(spec[2], spec[3])
        if source is not None and position is not None and not values.dropna().empty: optional_available.append(spec)
    if optional_available:
        with st.expander("Additional trends", expanded=False):
            for spec in optional_available: make_trend(*spec)

    with st.expander("View technical/raw data", expanded=False):
        st.json(selected.to_dict())
        st.dataframe(stations.drop(columns=["__position_pct"], errors="ignore"), hide_index=True, use_container_width=True)
    st.markdown("**Chat context**"); st.caption(f"Questions use uploaded values for selected strut {current}.")
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]): st.markdown(message["content"])
    question = st.chat_input(f"Ask about strut {current} or the defect summary")
    if question:
        messages = st.session_state.setdefault("chat_messages", [])
        messages.append({"role": "user", "content": question})
        try:
            answer = gemini_chat_response(question, selected, stations, messages)
        except Exception as exc:
            answer = f"Gemini chat is unavailable: {exc}"
        messages.append({"role": "assistant", "content": answer})
        st.rerun()

st.markdown("""
<style>
.stApp{background:#0b1120}[data-testid="stSidebar"]{background:#111827;border-right:1px solid #293750}.metric-card{background:#172033;border:1px solid #293750;border-radius:12px;padding:14px 16px;min-height:76px}.metric-label{color:#94a3b8;font-size:.76rem;text-transform:uppercase;letter-spacing:.06em}.metric-value{color:#f8fafc;font-size:1.15rem;font-weight:650;margin-top:7px}.flow-stage{background:#172033;border:1px solid #334155;border-radius:12px;padding:18px;text-align:center}.flow-number{color:#38bdf8;font-size:.72rem}.flow-title{color:#f8fafc;font-weight:700;margin-top:8px}.flow-description{color:#cbd5e1;padding:18px 4px}.flow-arrow{color:#38bdf8;font-size:2rem;text-align:center;padding-top:18px}
[data-testid="stSidebar"] [data-testid="stButton"] button{justify-content:flex-start;border-radius:7px;font-weight:600;min-height:42px;margin:2px 0}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]{background:transparent;border-color:transparent;color:#e5e7eb}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover{background:#1f2937;border-color:transparent;color:#fff}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]{background:#1f2937;border-color:#1f2937;border-left:3px solid #38bdf8;color:#f8fafc}
</style>
""", unsafe_allow_html=True)
summary,station,error=data()
with st.sidebar:
    st.caption("Navigation")
    st.session_state.setdefault("page", "Overview")
    nav_items = [("Overview", ":material/dashboard:"), ("Inspection Workflow", ":material/account_tree:"), ("Strut Analysis", ":material/analytics:")]
    for label, icon in nav_items:
        if st.button(label, icon=icon, key=f"nav_{label}", type="primary" if st.session_state.page == label else "secondary", use_container_width=True):
            st.session_state.page = label
            st.rerun()
page = st.session_state.page
if error: st.error(f"Could not read the uploaded CSVs: {error}")
if page=="Overview": overview(summary,station)
elif page=="Inspection Workflow": flowchart()
else: analysis(summary,station)