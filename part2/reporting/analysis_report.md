# Stage 3 Analysis Report

## Scope

Stage 2a analyzed `data\missing_struts\tif_stacks\210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif` using a bounded page-wise sample and produced **572 candidate defects**.
Stage 2b reviewed the 20 highest-priority candidates with local TIFF crops and PyVista off-screen renders.

These are candidate defects, not confirmed manufacturing defects. Visual agreement with the CT data does not establish manufacturing cause, metrological accuracy, or probability of detection.

## Stage 2a summary

- Candidate records: **572**
- Input shape: `[761, 815, 837]` (z, y, x)
- Segmentation threshold: `40264.0` (Otsu on bounded page-wise strided sample)
- Downsample factor: `4`
- Candidate type: `possible_missing_or_broken_strut`
- Candidate locations are in the registered CT voxel coordinate system.

## Stage 2b reviewed candidates

The `support` column is the v2 field `visual_validation.material_fraction_in_crop`; it is not `raw_local_support_fraction`.

| ID | Type | Status / assessment | Confidence | Location (x, y, z) | Boundary | Support | Representative views |
|---|---|---|---:|---|---|---:|---|
| strut-00010 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.542 | 79.04, 52.25, 85.71 | clear | 0.000488 | [iso](part2/visual_reasoner/outputs/strut-00010/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00010/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00010/side.png) |
| strut-00200 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.21, 68.37, 46.74 | clear | 0.047266 | [iso](part2/visual_reasoner/outputs/strut-00200/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00200/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00200/side.png) |
| strut-00201 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.41, 107.86, 46.63 | clear | 0.065283 | [iso](part2/visual_reasoner/outputs/strut-00201/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00201/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00201/side.png) |
| strut-00202 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.18, 68.48, 86.23 | clear | 0.047559 | [iso](part2/visual_reasoner/outputs/strut-00202/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00202/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00202/side.png) |
| strut-00203 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.38, 107.96, 86.12 | clear | 0.043799 | [iso](part2/visual_reasoner/outputs/strut-00203/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00203/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00203/side.png) |
| strut-00214 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.619 | 750.54, 88.32, 86.16 | clear | 0.033008 | [iso](part2/visual_reasoner/outputs/strut-00214/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00214/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00214/side.png) |
| strut-00218 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.542 | 730.89, 108.17, 86.09 | clear | 0.074316 | [iso](part2/visual_reasoner/outputs/strut-00218/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00218/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00218/side.png) |
| strut-00420 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.61, 147.34, 46.52 | clear | 0.066016 | [iso](part2/visual_reasoner/outputs/strut-00420/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00420/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00420/side.png) |
| strut-00421 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.82, 186.83, 46.41 | clear | 0.081201 | [iso](part2/visual_reasoner/outputs/strut-00421/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00421/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00421/side.png) |
| strut-00422 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.58, 147.45, 86.01 | clear | 0.051221 | [iso](part2/visual_reasoner/outputs/strut-00422/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00422/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00422/side.png) |
| strut-00423 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.78, 186.94, 85.90 | clear | 0.043799 | [iso](part2/visual_reasoner/outputs/strut-00423/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00423/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00423/side.png) |
| strut-00640 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 771.02, 226.32, 46.30 | clear | 0.074658 | [iso](part2/visual_reasoner/outputs/strut-00640/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00640/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00640/side.png) |
| strut-00642 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 770.98, 226.43, 85.79 | clear | 0.051318 | [iso](part2/visual_reasoner/outputs/strut-00642/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00642/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00642/side.png) |
| strut-00643 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 771.19, 265.92, 85.68 | clear | 0.042334 | [iso](part2/visual_reasoner/outputs/strut-00643/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00643/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00643/side.png) |
| strut-00862 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.850 | 771.39, 305.41, 85.57 | clear | 0.052490 | [iso](part2/visual_reasoner/outputs/strut-00862/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00862/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00862/side.png) |
| strut-00863 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.542 | 771.59, 344.89, 85.46 | clear | 0.038184 | [iso](part2/visual_reasoner/outputs/strut-00863/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00863/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00863/side.png) |
| strut-00946 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.542 | 278.28, 406.64, 84.89 | clear | 0.012061 | [iso](part2/visual_reasoner/outputs/strut-00946/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-00946/front.png)<br>[side](part2/visual_reasoner/outputs/strut-00946/side.png) |
| strut-01310 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.542 | 713.17, 503.09, 65.23 | clear | 0.025244 | [iso](part2/visual_reasoner/outputs/strut-01310/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-01310/front.png)<br>[side](part2/visual_reasoner/outputs/strut-01310/side.png) |
| strut-01666 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.542 | 457.29, 662.41, 84.33 | clear | 0.029736 | [iso](part2/visual_reasoner/outputs/strut-01666/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-01666/front.png)<br>[side](part2/visual_reasoner/outputs/strut-01666/side.png) |
| strut-01728 | possible_missing_or_broken_strut | likely missing/broken candidate; not manufacturing-confirmed | 0.542 | 674.29, 621.76, 64.87 | clear | 0.020410 | [iso](part2/visual_reasoner/outputs/strut-01728/isometric.png)<br>[front](part2/visual_reasoner/outputs/strut-01728/front.png)<br>[side](part2/visual_reasoner/outputs/strut-01728/side.png) |

Boundary review: **0 of 20** reviewed candidates were boundary-affected. Boundary-affected candidates must be reported as inconclusive rather than confirmed defects.

## Limitations and recommended next steps

- Stage 2a uses downsampling and max pooling, which favor recall but can create false gaps or bridges.
- Stage 2a did not perform CAD-to-CT registration or machine-learning classification; the registered topology is an expected-lattice reference, not proof of a manufacturing defect.
- Stage 2b screenshots are local visual evidence only. Occlusion, segmentation threshold choice, partial-volume effects, and scan artifacts can mislead interpretation.
- The current review supports the label `likely missing/broken candidate` for the clear-crop cases, not a confirmed manufacturing defect diagnosis.
- Add strut-axis sampling, connected-component checks, local diameter estimates, and calibrated uncertainty before production use.
- Boundary and clipped-crop cases, if encountered on future scans, should be routed to manual review and labeled inconclusive.


## Quantitative lattice measurements

Using voxel size **58.09 × 58.09 × 58.09 µm**, the Stage 2a pooled mask has whole-bbox relative density **0.1702** and inward-trimmed relative density **0.1657** where available. The median distance-transform thickness estimate is **116.2 µm** versus the **350 µm** design thickness (ratio **0.332**).

Of 572 Stage 2a candidates, **553** are in the descriptive edge region and **19** in the center region. The nearest-neighbor ratio is **0.756**, classified as **clustered** under the Monte-Carlo reference. For the 20 Stage 2b-reviewed candidates, the corresponding edge/center counts are 20/0.

Plots: [thickness distribution](part2/reporting/thickness_distribution.png) and [spatial distribution](part2/reporting/spatial_distribution.png).

Limitations: these values use a factor-4 max-pooled mask (effective voxel size about 232.36 µm), so thin-strut thickness and local density are resolution-limited. The inward trim is only a proxy for excluding outer skins/walls; a registered CAD wall mask would be preferable. Spatial categories and clustering are descriptive, and candidate locations remain possible defects rather than confirmed manufacturing defects.
