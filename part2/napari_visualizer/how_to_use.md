# Dual-Window Strut Visualizer

This tool provides a high-performance, dual-window 3D inspection environment in Napari for triaging defected struts. It uses memory-mapping (`tifffile.memmap`) to handle large CT datasets with low RAM overhead, offering synchronized Macro (Full Lattice) and Micro (Isolated Strut) views. The Micro metadata panel loads strut summaries and station-level measurements from the Stage 3 defect-analysis exports.

## Features
*   **Dual-Window Sync:** Applying a set of IDs updates the Micro window with a 1x-resolution crop of the first selected strut.
*   **Multi-Strut Selection:** Several strut IDs can be highlighted at once in the full-lattice view, while the Micro dropdown switches between them.
*   **Categorical Color-Coding:** Nominal struts are rendered in solid Blue; defects (missing/partial) are rendered in solid Orange.
*   **Interactive 3D Picking:** `Shift + Click` centerlines in the 3D viewer to append them to the selection, update both views, and load metadata.
*   **Memory Efficient:** The CT volume is memory-mapped from disk. Only the 2x downsampled lattice and the specific `20x20x20` cropped voxel chunks are loaded into memory.

## 1. Environment Setup

This project uses a native Windows Python environment (`dssi_env_win`). 

First, create and activate a new virtual environment (Conda is recommended):
```bash
conda create -n dssi_env_win python=3.10
conda activate dssi_env_win

```

Install the required dependencies. Note that `napari[all]` is required to install the underlying Qt GUI framework:

```bash
pip install "napari[all]" tifffile pandas numpy magicgui

```

## 2. Data Requirements

You need two files to run the visualizer:

1. **3D CT Scan:** A 3D `.tif` or `.tiff` file.
2. **Centerlines Data:** `napari_centerlines.csv` containing pre-computed ZYX coordinates, bounding boxes, and inventory metadata.
3. **Defect Analysis Data:** By default, the visualizer reads `part2/stage_3_defect_analysis/output/station_export_20260727T193004Z/defect_analysis_by_strut.csv` and `defect_analysis_by_station.csv`.

## 3. Running the Visualizer

Execute `visualize_struts.py` from your Windows terminal (PowerShell or Command Prompt).

**Important Note on WSL Paths:**
If your data resides in a WSL (Windows Subsystem for Linux) directory, but you are running this native Windows Python environment, you must pass the network path using the `\\wsl.localhost` prefix.

**Example Usage:**

```powershell
python C:\Users\Enduser\llnl_data_science_challenge_2026\stage3_visualization\visualize_struts.py 
  --scan "C:\Users\Enduser\llnl_data_science_challenge_2026\data\missing_struts\tif_stacks\210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" `
  --centerlines "C:\Users\Enduser\llnl_data_science_challenge_2026\part2\napari_visualizer\napari_centerlines.csv"

```

## 4. Controls & Navigation

* **Pan/Rotate:** Left-click and drag in the viewer.
* **Zoom:** Scroll wheel.
* **Select Struts (UI):** Enter comma- or space-separated IDs in the `Strut IDs` field in the Full Lattice window, then click `Apply`.
* **Select Struts (3D):** Hold `Shift` and `Left-Click` centerlines in the Full Lattice window to append them to the selection and apply it.
* **Inspect a Selected Strut:** Use the `Selected Strut` dropdown in the individual-strut window to switch among the applied IDs.
* **Clear Selection:** Click `Clear` in the Full Lattice window to remove all yellow highlights.

The defect-analysis CSV locations can be overridden with `--defect-by-strut` and `--defect-by-station` when running the visualizer.
