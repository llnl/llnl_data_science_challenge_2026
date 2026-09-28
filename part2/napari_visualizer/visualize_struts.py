import argparse
import os
import queue
import threading
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import tifffile
from magicgui.widgets import ComboBox, Container, Label, LineEdit, PushButton, TextEdit


SELECTED_STRUT_COLOR = "yellow"
SELECTED_STRUT_WIDTH = 5
TRANSPARENT_STRUT_COLOR = "transparent"


def parse_strut_ids(value):
    """Parse a comma- or whitespace-separated strut-ID entry.

    IDs are returned in first-entry order with duplicates removed. The parser
    deliberately returns strings because the CSV may contain IDs represented
    as either integers or strings by pandas.
    """
    tokens = str(value or "").replace(",", " ").split()
    result = []
    seen = set()
    for token in tokens:
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result


def format_strut_metadata(strut_row, geometry_row):
    """Format concise identity, defect, and structural metadata for Napari."""

    def value(*columns):
        for column in columns:
            for row in (strut_row, geometry_row):
                if row is not None and column in row.index and pd.notna(row[column]):
                    return row[column]
        return "Unavailable"

    lines = [
        "=== STRUT SUMMARY ===",
        f"Strut ID: {value('strut_id')}",
        f"Classification: {value('stage2_classification', 'inventory_classification')}",
        f"Primary defect: {value('primary_defect')}",
        f"Needs review: {value('needs_review')}",
        f"Expected by CAD: {value('expected_by_0point5_cad', 'inventory_expected_by_0point5_cad')}",
        "",
        "=== STRUCTURE ===",
        f"Unit cell: {value('unit_cell_ids', 'inventory_unit_cell_ids')}",
        f"Junctions: {value('junction0_id', 'inventory_junction0_id')} -> "
        f"{value('junction1_id', 'inventory_junction1_id')}",
        f"Junction degrees: {value('junction0_degree', 'inventory_junction0_degree')} -> "
        f"{value('junction1_degree', 'inventory_junction1_degree')}",
        f"Length (um): {value('inventory_length_um', 'length_um_from_centerline')}",
        f"Start (vox ZYX): {value('start_z_vox')}, {value('start_y_vox')}, {value('start_x_vox')}",
        f"End (vox ZYX): {value('end_z_vox')}, {value('end_y_vox')}, {value('end_x_vox')}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Dual-Window Strut Visualizer")
    parser.add_argument("--scan", type=str, required=True, help="Path to the 3D TIFF scan")
    parser.add_argument("--centerlines", type=str, required=True, help="Path to napari_centerlines.csv")
    parser.add_argument(
        "--unit-cell-padding-vox",
        type=int,
        default=10,
        help="Extra source voxels around the selected unit-cell crop (default: 10)",
    )
    parser.add_argument(
        "--macro-zoom",
        type=float,
        default=6.0,
        help="Full-lattice camera zoom after selecting a strut (default: 6)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("NAPARI_API_URL"),
        help="Optional FastAPI base URL for receiving dashboard selection events",
    )
    default_defect_dir = (
        Path(__file__).resolve().parents[2]
        / "part2/stage_3_defect_analysis/output/station_export_20260727T193004Z"
    )
    parser.add_argument(
        "--defect-by-strut",
        type=Path,
        default=default_defect_dir / "defect_analysis_by_strut.csv",
        help="Defect-analysis strut summary CSV",
    )
    parser.add_argument(
        "--defect-by-station",
        type=Path,
        default=default_defect_dir / "defect_analysis_by_station.csv",
        help="Defect-analysis station detail CSV",
    )
    args = parser.parse_args()

    if args.unit_cell_padding_vox < 0:
        parser.error("--unit-cell-padding-vox must be non-negative")
    if args.macro_zoom <= 0:
        parser.error("--macro-zoom must be greater than zero")

    print(f"Loading defect-analysis strut metadata from {args.defect_by_strut}...")
    defect_by_strut = pd.read_csv(args.defect_by_strut)
    if "strut_id" not in defect_by_strut.columns:
        raise ValueError(f"Defect strut CSV must contain strut_id: {args.defect_by_strut}")
    defect_strut_index = {
        str(strut_id): row
        for strut_id, row in defect_by_strut.set_index("strut_id").iterrows()
    }

    print(f"Loading defect-analysis station metadata from {args.defect_by_station}...")
    defect_by_station = pd.read_csv(args.defect_by_station)
    if "strut_id" not in defect_by_station.columns:
        raise ValueError(f"Defect station CSV must contain strut_id: {args.defect_by_station}")
    defect_station_index = {
        str(strut_id): rows.drop(columns=["strut_id"])
        for strut_id, rows in defect_by_station.groupby("strut_id", sort=False)
    }

    print(f"Loading TIFF from {args.scan} (memory-mapped)...")
    volume = tifffile.memmap(args.scan)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D TIFF scan, got shape {volume.shape}")

    print("Downsampling volume 2x for full-lattice rendering...")
    volume_2x = volume[::2, ::2, ::2]

    print(f"Loading centerlines from {args.centerlines}...")
    table = pd.read_csv(args.centerlines)
    id_col = "strut_id"
    unit_cell_col = "inventory_unit_cell_ids"
    required_columns = {
        id_col,
        unit_cell_col,
        "inventory_classification",
        "start_z_vox", "start_y_vox", "start_x_vox",
        "direction_z_vox", "direction_y_vox", "direction_x_vox",
        "center_z_vox", "center_y_vox", "center_x_vox",
        "bbox_z_min_inclusive", "bbox_y_min_inclusive", "bbox_x_min_inclusive",
        "bbox_z_max_exclusive", "bbox_y_max_exclusive", "bbox_x_max_exclusive",
    }
    missing_columns = sorted(required_columns.difference(table.columns))
    if missing_columns:
        raise ValueError("Centerlines CSV is missing required columns: " + ", ".join(missing_columns))
    if table[id_col].duplicated().any():
        raise ValueError("Centerlines CSV must contain one row per strut_id")

    def get_color(cls_val):
        if pd.isna(cls_val):
            return "magenta"
        val = str(cls_val).lower()
        if "missing" in val:
            return "red"
        if "partial" in val:
            return "orange"
        if "nominal" in val:
            return "cyan"
        return "yellow"

    table["custom_display_color"] = table["inventory_classification"].apply(get_color)
    base_colors = table["custom_display_color"].tolist()
    # Keep the widget/API representation stable even when pandas infers a
    # numeric dtype for the CSV column.
    strut_ids = [str(value) for value in table[id_col]]
    strut_index = {strut_id: index for index, strut_id in enumerate(strut_ids)}

    # The inventory currently assigns one unit-cell ID per strut. Treat missing
    # IDs as a single-strut fallback so a malformed row remains inspectable.
    unit_cell_to_indices = {}
    for index, unit_cell_id in enumerate(table[unit_cell_col]):
        if pd.notna(unit_cell_id) and str(unit_cell_id).strip():
            unit_cell_to_indices.setdefault(str(unit_cell_id), []).append(index)

    origin = table[["start_z_vox", "start_y_vox", "start_x_vox"]].to_numpy()
    direction = table[["direction_z_vox", "direction_y_vox", "direction_x_vox"]].to_numpy()
    vectors = np.stack((origin, direction), axis=1)
    bbox_min_columns = ["bbox_z_min_inclusive", "bbox_y_min_inclusive", "bbox_x_min_inclusive"]
    bbox_max_columns = ["bbox_z_max_exclusive", "bbox_y_max_exclusive", "bbox_x_max_exclusive"]
    volume_shape = np.asarray(volume.shape, dtype=int)

    print("Initializing Dual Napari Viewers...")
    viewer_full = napari.Viewer(title="Full Lattice (Macro)", ndisplay=3)
    viewer_iso = napari.Viewer(title="Selected Unit Cell (Micro)", ndisplay=3)

    viewer_full.add_image(
        volume_2x,
        name="Full CT Scan (2x)",
        colormap="gray",
        rendering="mip",
        depiction="volume",
        scale=(2, 2, 2),
    )
    full_vectors_layer = viewer_full.add_vectors(
        vectors,
        name="All Centerlines",
        properties=table,
        edge_color=base_colors,
        edge_width=2,
    )
    selected_full_layer = viewer_full.add_vectors(
        np.zeros((0, 2, 3)),
        name="Selected Centerline",
        edge_color=SELECTED_STRUT_COLOR,
        edge_width=SELECTED_STRUT_WIDTH,
    )

    isolated_ct_layer = viewer_iso.add_image(
        np.zeros((1, 1, 1)),
        name="Unit Cell 3D CT",
        colormap="gray",
        rendering="mip",
        depiction="volume",
    )
    isolated_context_layer = viewer_iso.add_vectors(
        np.zeros((0, 2, 3)),
        name="Unit Cell Centerlines",
        edge_color=["gray"],
        edge_width=2,
    )
    isolated_vector_layer = viewer_iso.add_vectors(
        np.zeros((0, 2, 3)),
        name="Selected Centerline",
        edge_color=SELECTED_STRUT_COLOR,
        edge_width=SELECTED_STRUT_WIDTH,
    )

    selected_ids_input = LineEdit(label="Strut IDs:", value="")
    apply_button = PushButton(text="Apply")
    clear_button = PushButton(text="Clear")
    selection_status = Label(value="No struts selected")
    dropdown = ComboBox(choices=[], label="Selected Strut:")
    metadata_display = TextEdit(label="Metadata:", value="Select a strut...")

    def context_indices_for(index):
        unit_cell_id = table.iloc[index][unit_cell_col]
        if pd.isna(unit_cell_id) or not str(unit_cell_id).strip():
            return [index], None
        return unit_cell_to_indices.get(str(unit_cell_id), [index]), str(unit_cell_id)

    def context_bounds(indices):
        cell_rows = table.iloc[indices]
        lower = cell_rows[bbox_min_columns].to_numpy(dtype=int).min(axis=0)
        upper = cell_rows[bbox_max_columns].to_numpy(dtype=int).max(axis=0)
        lower = np.maximum(lower - args.unit_cell_padding_vox, 0)
        upper = np.minimum(upper + args.unit_cell_padding_vox, volume_shape)
        if np.any(lower >= upper):
            raise ValueError("Selected strut's context crop does not intersect the scan")
        return lower, upper

    selected_indices = []

    def update_micro_view(strut_id, recenter_macro=True):
        index = strut_index[str(strut_id)]
        row = table.iloc[index]
        context_indices, unit_cell_id = context_indices_for(index)
        lower, upper = context_bounds(context_indices)

        selected_id = str(strut_id)
        defect_summary = defect_strut_index.get(selected_id)
        station_rows = defect_station_index.get(selected_id)
        info = format_strut_metadata(defect_summary, row)
        info += (
            "\n\n=== VISUALIZER CONTEXT ===\n"
            f"Unit Cell: {unit_cell_id if unit_cell_id is not None else 'Unavailable'} "
            f"({len(context_indices)} struts)"
        )
        metadata_display.value = info

        z_min, y_min, x_min = lower
        z_max, y_max, x_max = upper
        crop_data = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        isolated_ct_layer.data = crop_data
        isolated_ct_layer.translate = tuple(lower)
        c_min, c_max = float(np.min(crop_data)), float(np.max(crop_data))
        if c_min < c_max:
            isolated_ct_layer.contrast_limits = (c_min, c_max)

        isolated_context_layer.data = vectors[context_indices]
        # The selected strut is drawn separately as the yellow highlight, so
        # hide its classification-colored copy in the unit-cell context layer.
        isolated_context_layer.edge_color = [
            TRANSPARENT_STRUT_COLOR if i == index else base_colors[i]
            for i in context_indices
        ]
        isolated_vector_layer.data = vectors[index:index + 1]

        if recenter_macro:
            selected_center = tuple(
                row[["center_z_vox", "center_y_vox", "center_x_vox"]].to_numpy(dtype=float)
            )
            viewer_full.camera.center = selected_center
            viewer_full.camera.zoom = args.macro_zoom

        context_center = tuple((lower + upper - 1) / 2)
        viewer_iso.reset_view()
        viewer_iso.camera.center = context_center

    def update_ui(*_):
        if dropdown.value is not None:
            update_micro_view(dropdown.value)

    def apply_selection(*_):
        nonlocal selected_indices
        requested_ids = parse_strut_ids(selected_ids_input.value)
        unknown_ids = [strut_id for strut_id in requested_ids if strut_id not in strut_index]
        if unknown_ids:
            selection_status.value = "Unknown strut ID(s): " + ", ".join(unknown_ids)
            return
        if not requested_ids:
            selection_status.value = "Enter at least one strut ID"
            return

        selected_indices = [strut_index[strut_id] for strut_id in requested_ids]
        selected_ids_input.value = ", ".join(requested_ids)
        highlighted_colors = list(base_colors)
        for index in selected_indices:
            highlighted_colors[index] = TRANSPARENT_STRUT_COLOR
        full_vectors_layer.edge_color = highlighted_colors
        full_vectors_layer.selected_data = set(selected_indices)
        selected_full_layer.data = vectors[selected_indices]

        dropdown.choices = requested_ids
        dropdown.value = requested_ids[0]
        selection_status.value = f"{len(requested_ids)} strut(s) selected"
        print(f"Selected strut(s): {', '.join(requested_ids)}", flush=True)
        # The first selected ID determines the initial camera position.
        update_micro_view(requested_ids[0])

    def clear_selection(*_):
        nonlocal selected_indices
        selected_indices = []
        selected_ids_input.value = ""
        full_vectors_layer.edge_color = list(base_colors)
        full_vectors_layer.selected_data = set()
        selected_full_layer.data = np.zeros((0, 2, 3))
        dropdown.choices = []
        isolated_context_layer.data = np.zeros((0, 2, 3))
        isolated_vector_layer.data = np.zeros((0, 2, 3))
        metadata_display.value = "Select a strut..."
        selection_status.value = "No struts selected"

    def start_api_selection_client():
        """Optionally receive dashboard selections without coupling standalone Napari to FastAPI."""
        if not args.api_url:
            return None
        try:
            import websocket
            from qtpy.QtCore import QTimer
        except ImportError as exc:
            selection_status.value = f"API client unavailable: install websocket-client ({exc})"
            return None

        incoming_selections = queue.Queue()
        websocket_url = args.api_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ws"

        def receive_events():
            try:
                connection = websocket.create_connection(websocket_url, timeout=5)
                # Use the timeout only while establishing the connection. An
                # idle dashboard should not be treated as a disconnected API.
                connection.settimeout(None)
                print(f"API connected: {websocket_url}", flush=True)
                while True:
                    raw_event = connection.recv()
                    print(f"Received API prompt: {raw_event}", flush=True)
                    incoming_selections.put(raw_event)
            except Exception as exc:
                print(f"API connection error: {exc}", flush=True)
                incoming_selections.put({"event_type": "API_ERROR", "error": str(exc)})

        def apply_events():
            import json

            while True:
                try:
                    raw_event = incoming_selections.get_nowait()
                except queue.Empty:
                    return
                if isinstance(raw_event, dict):
                    event = raw_event
                else:
                    try:
                        event = json.loads(raw_event)
                    except (TypeError, ValueError):
                        continue
                if event.get("event_type") != "STRUTS_SELECTED":
                    if event.get("event_type") == "API_ERROR":
                        selection_status.value = "API connection closed"
                    continue
                ids = event.get("data", {}).get("strut_ids", [])
                if ids:
                    selected_ids_input.value = ", ".join(str(value) for value in ids)
                    apply_selection()

        thread = threading.Thread(target=receive_events, daemon=True, name="napari-api-selection")
        thread.start()
        timer = QTimer()
        timer.timeout.connect(apply_events)
        timer.start(100)
        selection_status.value = f"Listening for dashboard selections at {websocket_url}"
        return timer

    dropdown.changed.connect(update_ui)
    apply_button.clicked.connect(apply_selection)
    clear_button.clicked.connect(clear_selection)

    @full_vectors_layer.mouse_drag_callbacks.append
    def on_3d_click(layer, event):
        if "Shift" not in event.modifiers:
            return
        clicked_index = layer.get_value(
            event.position,
            world=True,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
        )
        if clicked_index is not None and isinstance(clicked_index, (int, np.integer)):
            clicked_id = str(table.iloc[int(clicked_index)][id_col])
            current_ids = parse_strut_ids(selected_ids_input.value)
            if clicked_id not in current_ids:
                current_ids.append(clicked_id)
            selected_ids_input.value = ", ".join(current_ids)
            apply_selection()

    full_ui_container = Container(
        widgets=[selected_ids_input, apply_button, clear_button, selection_status]
    )
    viewer_full.window.add_dock_widget(full_ui_container, name="Strut Selection", area="right")
    micro_ui_container = Container(widgets=[dropdown, metadata_display])
    viewer_iso.window.add_dock_widget(micro_ui_container, name="Strut Inspector", area="right")

    api_timer = start_api_selection_client()

    print("Launching windows...")
    napari.run()


if __name__ == "__main__":
    main()
