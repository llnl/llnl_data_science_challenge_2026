# Stage 3: registration-aware strut visualization

This module maps a `strut_id` from the registration JSON to its two registered
junctions, converts registration XYZ coordinates to TIFF ZYX indexing, and
extracts a bounded CT crop. It writes a 3-D PNG render and traceable coordinate
metadata; the crop and thresholded mask are kept in memory and are not written
as TIFF files.

From the repository root, for the included data:

```bash
python -m stage3_visualization.strut_visualizer \
  --scan "part2/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" \
  --registration "part2/data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json" \
  --strut-id 9 \
  --csv part2/stage_2a_developer_output/all_struts_inventory.csv
```

Use `--threshold` to change the CT segmentation cutoff and `--margin-voxels`
to enlarge or reduce the extracted region. The default border is 10 source
voxels and applies to the CT crop, binary mask, and 3-D PNG render. The
visualizer records and labels the complete registered endpoint-to-endpoint
centerline length in source voxels. Rendering defaults to `--downsample 1` so
the full centerline is shown at source-voxel scale; the included lattice
struts are approximately 55.85 source voxels long (about 3.24 mm), with the
10-voxel border providing at least about 60 source voxels across the long
crop axis. The default threshold is 40000 for the included uint16 scan; it
should be adjusted for
scans with a different intensity scale.

## Interactive Napari graph viewer

`visualize_struts.py` reads the precomputed Napari centerline CSV, colors
centerlines by classification, and opens a macro lattice view alongside a
selected unit-cell CT view:

```bash
python stage3_visualization/visualize_struts.py \
  --scan "part2/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" \
  --centerlines part2/napari_visualizer/napari_centerlines.csv
```

Select a strut from the Inspector dropdown or Shift-click its centerline in
the macro view. The selected strut becomes a wide yellow line, the macro
camera zooms to it, and the micro view loads all struts assigned to its unit
cell with CT context. Use `--unit-cell-padding-vox` to add border voxels to
that crop and `--macro-zoom` to tune the focused macro camera level.
