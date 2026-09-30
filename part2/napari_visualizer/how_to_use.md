# Dual-Window Strut Visualizer

This tool provides a high-performance, dual-window 3D inspection environment in Napari for triaging defected struts. It uses memory-mapping (`tifffile.memmap`) to handle large CT datasets with low RAM overhead, offering synchronized Macro (Full Lattice) and Micro (Isolated Strut) views.

## Features
*   **Dual-Window Sync:** Navigating in the Macro window automatically updates the Micro window with a 1x-resolution crop of the selected strut.
*   **Categorical Color-Coding:** Nominal struts are rendered in solid Blue; defects (missing/partial) are rendered in solid Orange.
*   **Interactive 3D Picking:** `Shift + Click` any strut in the 3D viewer to snap both cameras to it and load its metadata.
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
* **Select Strut (3D):** Hold `Shift` and `Left-Click` a centerline in the Full Lattice window.
* **Select Strut (UI):** Use the dropdown menu in the Inspector pane on the right.