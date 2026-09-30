import argparse

import napari
import numpy as np
import pandas as pd
import tifffile
from magicgui.widgets import ComboBox, Container, TextEdit


SELECTED_STRUT_COLOR = "yellow"
SELECTED_STRUT_WIDTH = 5
TRANSPARENT_STRUT_COLOR = "transparent"


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
    args = parser.parse_args()

    if args.unit_cell_padding_vox < 0:
        parser.error("--unit-cell-padding-vox must be non-negative")
    if args.macro_zoom <= 0:
        parser.error("--macro-zoom must be greater than zero")

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
    strut_index = {strut_id: index for index, strut_id in enumerate(table[id_col])}

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
        np.zeros((1, 2, 3)),
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
        np.zeros((1, 2, 3)),
        name="Unit Cell Centerlines",
        edge_color=["gray"],
        edge_width=2,
    )
    isolated_vector_layer = viewer_iso.add_vectors(
        np.zeros((1, 2, 3)),
        name="Selected Centerline",
        edge_color=SELECTED_STRUT_COLOR,
        edge_width=SELECTED_STRUT_WIDTH,
    )

    strut_choices = table[id_col].tolist()
    dropdown = ComboBox(choices=strut_choices, label="Select Strut:")
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

    def update_ui(*_):
        strut_id = dropdown.value
        index = strut_index[strut_id]
        row = table.iloc[index]
        context_indices, unit_cell_id = context_indices_for(index)
        lower, upper = context_bounds(context_indices)

        info = (
            f"--- STRUT {row['strut_id']} ---\n"
            f"Classification: {row.get('inventory_classification', 'N/A')}\n"
            f"Unit Cell: {unit_cell_id if unit_cell_id is not None else 'Unavailable'} "
            f"({len(context_indices)} struts)\n"
            f"Material Occupancy: {row.get('inventory_ct_material_occupancy', 0):.2f}\n"
            f"Length (µm): {row.get('length_um_from_centerline', 0):.2f}\n"
            f"Missing Vol (mm³): {row.get('inventory_estimated_missing_volume_mm3', 0):.6f}"
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

        highlighted_colors = list(base_colors)
        # Keep the full-lattice base layer from showing through the separate
        # yellow selected-centerline overlay.
        highlighted_colors[index] = TRANSPARENT_STRUT_COLOR
        full_vectors_layer.edge_color = highlighted_colors
        full_vectors_layer.selected_data = {index}
        selected_full_layer.data = vectors[index:index + 1]

        selected_center = tuple(
            row[["center_z_vox", "center_y_vox", "center_x_vox"]].to_numpy(dtype=float)
        )
        context_center = tuple((lower + upper - 1) / 2)
        viewer_full.camera.center = selected_center
        viewer_full.camera.zoom = args.macro_zoom
        viewer_iso.reset_view()
        viewer_iso.camera.center = context_center

    dropdown.changed.connect(update_ui)

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
            dropdown.value = table.iloc[int(clicked_index)][id_col]

    ui_container = Container(widgets=[dropdown, metadata_display])
    viewer_full.window.add_dock_widget(ui_container, name="Inspector", area="right")

    print("Launching windows...")
    napari.run()


if __name__ == "__main__":
    main()
