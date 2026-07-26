# Strut flagging criteria

How `src/strut_cylinder_segmentation.py` decides that a nominal strut is a
missing candidate. Line references are to that file.

## The criterion

One threshold on one derived quantity (`:428-429`):

```python
cut = args.area_fraction * NOMINAL_CROSS_SECTION_VOXELS   # 0.25 × 12.26 = 3.06 vox²
missing_mask = area < cut
```

A strut is flagged when it carries **less than 25% of its nominal design
cross-section**.

## How `area` is computed

### 1. Segment the volume once (`:382-383`)

```python
otsu_threshold = threshold_otsu(volume)      # 40049 for this scan
segmentation = volume >= otsu_threshold      # 11.30% of voxels are material
```

This is the only place the Otsu threshold is used, and it acts on **voxel
intensities**.

### 2. Rasterize a cylinder along each strut (`cylinder_mask`, `:139-146`)

Endpoints `p0`, `p1` are the strut's two junctions from the registered JSON,
`u` is the unit vector along the strut, `L` its length, and `d` the offset of a
candidate voxel from `p0`. A voxel is inside when both hold:

```
radial = |d - (d·u)u|  <=  radius
end_trim·L  <=  d·u  <=  (1 - end_trim)·L
```

The search is confined to the bounding box of the endpoints expanded by
`radius` and clipped to the volume.

### 3. Count and normalize (`:169-180`)

```python
sampled_lengths = ‖end - start‖ * (1 - 2*end_trim)
area = n_segmented / sampled_lengths
```

Dividing by the length *actually sampled* makes `area` a mean cross-section in
vox², so it stays comparable across both radius and trim settings.

## Where the nominal cross-section comes from

`NOMINAL_CROSS_SECTION_VOXELS = π·(0.1·39.5/2)² = 12.26 vox²` (`:79`). It is
built from two inputs, one read from the data and one inferred.

### Thickness: read from the JSON

Every strut in the lattice JSON carries `thickness: 0.1`, in design units. The
set of distinct values across all 18468 struts is exactly `{0.1}`, so this is a
single global constant rather than a per-strut property.

### Scale: inferred from strut lengths

No file states voxels-per-design-unit, so it is derived from the ratio of
registered to nominal strut length. Both JSONs list the same struts in the same
id order:

```
nominal length    = 1.4142 design units   (a √2 face diagonal, identical for all struts)
registered length = 55.846 voxels         (identical for all struts)
scale             = 55.846 / 1.4142 = 39.4888 voxels per design unit
```

The ratio is the same for every strut (min = max), as expected for a similarity
registration. The script hardcodes this as `39.5`.

### Conversion

Reading `thickness` as a **diameter**:

```
diameter = 0.1 × 39.5 = 3.95 voxels
area     = π × (3.95 / 2)² = 12.25 vox²
```

## Caveats on the nominal cross-section

Both of the following move the cut, so they are worth knowing before trusting a
flagged count.

**Diameter-vs-radius is an inference, not stated in the JSON.** If `thickness`
were a radius, the nominal area would be 48.99 vox² — exactly 4× larger — and
the 25% cut would move from 3.06 to 12.25 vox². Two observations support the
diameter reading:

- The measured median area at radius 3 was 14.6 vox², close to 12.25 and nowhere
  near 49.
- A strut of radius 3.95 could not fit inside a radius-3 cylinder at all, yet
  those cylinders were plainly capturing whole struts.

**The 39.5 scale is a hardcoded literal.** The true ratio is 39.4888, so the
constant carries ~0.03% error, which is negligible here. The real hazard is that
it is fixed: on a lattice with different unit-cell spacing, or a scan at
different resolution, it would silently produce a wrong cut. Computing it at
runtime as `registered_length / nominal_length` — both already loaded — would
remove that failure mode.

## Related but separate statistic

`empty_mask = stats["n_segmented"] == 0` (`:431`) counts cylinders containing no
material at all. It is reported alongside the flag count and is a strict subset
of the flagged set, but it is **not** the flagging criterion.

## Parameters that materially change the result

| Parameter | Default | Effect |
|---|---|---|
| `--area-fraction` | 0.25 | The cut, as a fraction of nominal cross-section. |
| `--end-trim` | **0.0** | Fraction of the span dropped at each end. |
| `--radii` | 2 3 5 7 8 | Cylinder radii to sweep. |
| `--keep-bottom-layer` | off | Include the machined-off bottom face. |

Two of these are not optional in practice:

- **`--end-trim` must be non-zero.** Twelve struts meet at a junction, and that
  blob stays bright when any one of them is absent, so a full-span cylinder can
  never register an interior strut as empty. With the bottom face excluded and
  no trim, the detector found 1 flagged strut in 17460. `--end-trim 0.25`
  restores sensitivity.
- **Radius must be ≥7** to absorb the ~0.8% x-scale registration drift in the
  `registered_jsons/` coordinates (~5.6 voxels across the specimen, against
  struts only ~4 voxels thick).

Because `end_trim` defaults to 0.0, reproducing the headline result requires
passing it explicitly:

```bash
python src/strut_cylinder_segmentation.py \
    --radii 8 --tiff-radius 8 --end-trim 0.25 --suffix _nobottom_trim25
```

That configuration gives 95 struts flagged (0.54%) and 87 fully empty (0.50%),
against a 0.5% nominal of ~92 struts.
