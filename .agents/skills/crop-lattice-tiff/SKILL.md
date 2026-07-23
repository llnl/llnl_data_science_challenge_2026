---
name: crop-lattice-tiff
description: Crop border/background regions from a 3D CT lattice TIFF using a reproducible threshold-defined foreground bounding box, then regenerate a skeleton wireframe. Use when trimming empty CT scan borders while preserving the full lattice-cell extent, or when creating a compact TIFF and Crosshair-ready 3D skeleton visualization.
---

# Crop Lattice TIFF

Use the bundled scripts sequentially. They preserve source data and write every
derived artifact to explicit output paths.

## Workflow

1. Crop from the thresholded foreground extent. The crop includes the first and
   last non-background voxels on every axis; it does not remove partial cells.

   ```bash
   python scripts/crop_tiff_to_foreground.py INPUT.tif OUTPUT.tif \
     --threshold 50000 --metadata OUTPUT.crop.json
   ```

   When bright acquisition material occupies leading slices, set a documented
   axial start after inspecting the animation. For the 9×9×9 octet scan, use
   `--z-start 32` to omit slices 0–31. Use `--z-stop 732` to omit the trailing
   bright border (slices 732–760) from that same scan.

2. Skeletonize the cropped TIFF using the same threshold. This loads the
   cropped volume into memory, so ensure at least several times its raw size is
   available.

   ```bash
   python scripts/skeletonize_thresholded_tiff.py OUTPUT.tif OUTPUT.skeleton.npy \
     --threshold 50000
   ```

   If this cannot fit in memory, reuse a skeleton made from the *same source
   TIFF and threshold* by cropping it with the TIFF crop metadata. This is
   valid only when the TIFF crop removes background; record the fallback.

   ```bash
   python scripts/crop_skeleton_by_metadata.py SOURCE.skeleton.npy OUTPUT.crop.json \
     OUTPUT.skeleton.npy
   ```

3. Create continuous skeleton-branch paths for a Crosshair `scatter3d` line
   wireframe. Do not sample individual graph edges, which look like a point
   cloud at 3D viewing distance.

   ```bash
   python scripts/skeleton_to_branch_wireframe.py OUTPUT.skeleton.npy OUTPUT.parquet
   ```

4. Create a compact axial GIF animation from the cropped TIFF.

   ```bash
   python scripts/tiff_to_gif.py OUTPUT.tif OUTPUT.gif --frame-stride 10 --max-dimension 256
   ```

   For a Crosshair animation with Play/Pause controls and a slice slider, emit
   a Plotly animation specification instead:

   ```bash
   python scripts/tiff_to_plotly_animation_spec.py OUTPUT.tif OUTPUT.animation.json \
     --frame-stride 10 --max-dimension 128
   ```

## Validation

- Read the JSON metadata and verify `crop_start`/`crop_stop` and output shape.
- Confirm the cropped TIFF is `uint16` and the skeleton shape equals the cropped
  TIFF shape.
- Plot the Parquet columns `x`, `y`, and `z` as a Plotly `scatter3d` trace with
  `mode="lines"` and `connectgaps=false`.

## Notes

The threshold should match the segmentation workflow used to define material.
If the foreground touches a scan boundary, that axis is intentionally not
cropped: removing it would clip the lattice rather than just remove a border.
