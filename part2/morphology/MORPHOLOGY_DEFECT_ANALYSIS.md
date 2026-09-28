# Morphology defect analysis extension

`morphology_defect_analysis.py` is an additive Stage 2 analysis. It reads the
existing Stage 2a inventory, the existing Napari centerline table, the source
CT TIFF, and the four existing `*_mask_ds4.tif` masks. It writes only new files
with the `morphology_defect_test_` prefix.

The inventory remains authoritative for Stage 2a classification and existing
metrics. Napari supplies the registered CAD endpoint coordinates because the
inventory CSV does not contain endpoints. The analyzer samples stations along
each endpoint-to-endpoint line. Each station records CT material fraction, an
area-based effective-diameter proxy, and nearest values from the existing
masks. Per-strut results aggregate those station rows into provisional
`thin`, `thick`, `broken`, and `bent` screening labels.

The diameter is an effective cross-sectional proxy based on material fraction
inside a fixed transverse disk; it is not a replacement for a validated
metrology measurement. Mask values are sampled from the existing 4x
downsampled masks and are not used to rewrite Stage 2a classifications.

## Reproduce the 16-strut test

From the repository root:

```powershell
python part2/morphology/morphology_defect_analysis.py `
  --config part2/morphology/morphology_defect_thresholds.json `
  --max-struts 16
```

Outputs:

- `morphology_defect_test_summary.csv`: one row per analyzed strut.
- `morphology_defect_test_stations.csv`: one row per sampled station.
- `morphology_defect_test_provenance.json`: selected IDs, input paths, and policy.

The script refuses to replace any of these new test outputs. Delete or move
them only with explicit approval, or select a different output directory.
