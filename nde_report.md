# NDE Report: Unit-Cell Volume

## Summary

The analyzed dataset is the `unitcell` volume with its corresponding binary mask and skeleton. All three arrays are shape-compatible at **256 × 256 × 256**. The skeleton is one connected structure and is fully contained within the segmented mask.

## Feature metrics

| Data product | Shape | Key metrics |
|---|---:|---|
| Original volume (`data/unitcell/unitcell.npy`) | 256 × 256 × 256 | 16,777,216 voxels; mean intensity 0.000539; standard deviation 0.002418; range −0.003129 to 0.015258 |
| Segmented mask (`data/unitcell/unitcell_mask.npy`) | 256 × 256 × 256 | 717,852 occupied voxels; 4.279% of the volume; mean intensity within mask 0.011696; within-mask standard deviation 0.001410 |
| Skeleton (`data/unitcell/unitcell_skeleton.npy`) | 256 × 256 × 256 | 3,182 occupied voxels; 0.443% of mask voxels; 1 connected component; 39 endpoints; 137 branch-point voxels; approximate graph length 4,605.1 voxel units |

## Visual gallery

The views below show the segmented 3D structure with the skeleton overlaid in red. Both use a threshold of 0.5 and a 2× downsample for rendering.

### View A — elevation 30°, azimuth 45°

![3D mask and skeleton, View A](nde_view_a.png)

### View B — elevation 60°, azimuth 45°

![3D mask and skeleton, View B](nde_view_b.png)

## Analysis

The mask occupies 4.279% of the full voxel grid and contains substantially higher intensity than the global volume mean (0.011696 versus 0.000539), consistent with the mask isolating the material structure from the surrounding field. Every skeleton voxel is inside the mask, giving 100% skeleton-to-mask containment and no detected out-of-mask skeleton artifacts. The single connected component indicates a continuous modeled network; its 39 endpoints and 137 branch-point voxels quantify the observed strut connectivity. The skeleton occupies only 0.443% of the mask by voxel count, as expected for a one-voxel-wide medial representation rather than a filled material volume.

## Processing notes

- Shape compatibility was checked before feature extraction.
- Mean intensity was computed both globally and over mask-positive voxels.
- Skeleton complexity was measured from 26-neighborhood voxel connectivity; branch points are skeleton voxels with at least three neighboring skeleton voxels.
- Approximate skeleton length sums the Euclidean distances between unique neighboring skeleton-voxel pairs.
