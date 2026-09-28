# Literature Review: Transform-aware Octet-Truss Defect Metrology

## Phase 1 — explored data structure

I inspected `registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json` directly. Its top level is three lists:

- `junctions`: 10,206 records. Each has an `id`, a three-coordinate `position`, and lattice `indices`.
- `struts`: 18,468 records. Each has an `id`, `junction0`, `junction1`, `unit_cell_edge_idx`, and nominal `thickness` (all listed as `0.1`).
- `unit_cells`: 729 records, indexed from `(0,0,0)` through `(8,8,8)`. Each lists 24 strut IDs.

The junction coordinates span approximately `[58.76, 48.57, 24.50]` to `[773.74, 764.94, 737.85]`, and every strut has length about `55.846` in these registered coordinates. Junction degrees range from 1 to 8, which is consistent with a finite lattice: boundary struts/junctions must not be judged with the same criteria as interior ones. These findings make an **edge-wise graph measurement** the central method. A global voxel-difference component could merge across junctions; instead, each graph edge defines a nominal strut tube, while unit-cell membership supports local spatial summaries.

The target is a large TIFF stack (about 1.04 GB) and `0.5.stl` (about 175 MB). The template directory contains paired OBJ/NPY simulations for the four requested classes and also nominal/thin controls. Use the templates to calibrate thresholds and decision limits before inspecting the target.

## Phase 2 — coordinate transformation

The data are mathematically registered but may be represented in different on-disk coordinate frames. Treat the registration JSON as the authoritative source of graph points and parse any supplied homogeneous transform matrix/voxel-origin/spacing fields without assuming their direction. For a point in mesh coordinates `p_m=[x,y,z,1]^T`, use:

```text
p_scan = T_mesh_to_scan p_m
index_(z,y,x) = ((p_scan - origin) / (s_x,s_y,s_z)) reordered to TIFF axes
```

If the metadata stores `T_scan_to_mesh`, invert it once with a condition-number check. Verify the result without optimizing alignment: transformed STL bounds and graph junction positions must lie in the scan ROI; sampled graph edges must pass through high-intensity material. Preserve the verified forward matrix, inverse, origin, axis permutation, and voxel pitch in provenance. `trimesh` supports 4×4 homogeneous transforms and transform-aware voxel grids ([Trimesh transform documentation](https://trimesh.org/trimesh.voxel.transforms.html)).

## Phase 3 — deterministic defect measurements

First segment CT material with a documented histogram/local threshold calibrated on the NPY controls. Then rasterize the transformed nominal STL/edge tubes onto the TIFF index grid. Construct directional disagreement masks:

```text
loss   = CAD_material AND NOT CT_material
excess = NOT CAD_material AND CT_material
```

Create a nominal signed-distance field (SDF) and suppress a calibrated surface band before labeling. The exact Euclidean distance transform accepts physical per-axis sampling ([SciPy EDT documentation](https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.ndimage.distance_transform_edt.html)); this converts surface tolerance into a transparent micrometre rule instead of a raw-XOR artifact.

| Defect | Method | Required evidence |
| --- | --- | --- |
| Bent | Compare segmented as-built centerline/surface to nominal edge axis using robust signed normal displacement and curvature change. | High occupancy but a persistent displacement above SDF/nominal-control tolerance; exclude node bands. |
| Broken | Find an interior `loss` component within a trimmed edge tube and measure its empty centerline run. | CT material on both sides of the gap. |
| Missing | Score every graph edge by tubular CT occupancy, CAD-loss volume, and longest contiguous empty centerline run. | Extended, stable interior absence after trimming both endpoint neighbourhoods. |
| Inflated | Retain `excess` material attached to a nominal strut and measure outward SDF/radius increase. | Connectedness to a strut; reject detached powder/speckle. |

Use 26-connectivity only after the CAD/SDF constraints. SciPy exposes the 3-D connectivity structure explicitly, including full diagonal connectivity ([SciPy morphology documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html)). The literature likewise identifies XCT as a suitable modality for lattice-strut shape-defect measurements ([Le Goïc et al., 2022](https://www.sciencedirect.com/science/article/abs/pii/S014163592200040X)).

## Memory management

Open TIFF data with `tifffile.memmap`; the library returns a memory-mapped array when TIFF storage permits it ([tifffile documentation](https://iridescent.ink/tifffile/tifffile.html)). Otherwise use Dask/Zarr chunks. Crop to the transformed lattice ROI and process overlapping Z slabs. Keep masks as `bool`/`uint8`; retain float data only for the current slab and an ROI-cropped SDF. Merge component labels touching adjacent overlap planes with a union-find table. Vectorize strut-point samples in batches; never iterate through all volume voxels in Python.

Avoid global dense mesh voxelization at target resolution. `trimesh.voxel.creation.local_voxelize` is specifically intended to reduce memory cost by voxelizing a local mesh neighbourhood ([Trimesh documentation](https://trimesh.org/trimesh.voxel.creation.html)); use it around each slab/edge group. Cache transformed mesh bounds and sparse tube indices. For a non-watertight mesh, prefer distance-to-surface tubes to a filled mesh test, because filled voxelization may be invalid.

## Physical scaling and limits

For calibrated TIFF pitch `(s_z,s_y,s_x)` in µm/voxel, use:

```text
voxel_volume = s_z*s_y*s_x                         [µm³]
component_volume = voxel_count*voxel_volume         [µm³]
physical_distance = sqrt((Δz*s_z)^2+(Δy*s_y)^2+(Δx*s_x)^2) [µm]
```

For an edge direction `u` and `n` index samples, gap length is `n*sqrt((u_z*s_z)^2+(u_y*s_y)^2+(u_x*s_x)^2)` µm. Supply that spacing to EDT routines to obtain SDF distances directly in µm. Do not derive physical units from STL coordinates until the STL unit convention and full transform chain are confirmed.

Report threshold/tube-radius/transform sensitivity. Principal false positives are transformed-axis mistakes, partial-volume surface voxels, untrimmed junction overlap, beam hardening/ring artifacts, loose powder, and features below the scan’s effective resolution. Published micro-CT AM guidance similarly stresses reproducible thresholding and the resolution dependence of defect measurement ([du Plessis et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6159003/)).

## References

- SciPy, [exact Euclidean distance transform](https://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.ndimage.distance_transform_edt.html) and [binary connectivity structures](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html).
- Trimesh, [voxel transforms](https://trimesh.org/trimesh.voxel.transforms.html) and [local voxelization](https://trimesh.org/trimesh.voxel.creation.html).
- tifffile, [`memmap`](https://iridescent.ink/tifffile/tifffile.html).
- A. du Plessis et al. (2018), [micro-CT AM porosity workflow](https://pmc.ncbi.nlm.nih.gov/articles/PMC6159003/).
- S. Le Goïc et al. (2022), [shape-defect analysis for AM lattice struts](https://www.sciencedirect.com/science/article/abs/pii/S014163592200040X).
