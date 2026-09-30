---
name: npy-metadata-extractor
description: Inspect .npy volumetric datasets and report their shape, data type, minimum, maximum, mean, and non-zero voxel count. Use when the user asks to inspect, summarize, or get metadata for one or more .npy files.
---

# NPY Metadata Workflow

Load the requested `.npy` file with NumPy.

Report:

- File path
- Array shape
- Data type
- Minimum and maximum values
- Mean value
- Number and percentage of non-zero voxels

Do not modify, overwrite, or create dataset files. If a requested file is missing, report its path and stop.