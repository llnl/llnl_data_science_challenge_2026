import tifffile
import trimesh
import numpy as np
from pathlib import Path

tif_file = Path("part2/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif")
stl_file = Path("part2/data/missing_struts/stls/0.5.stl")

print("1. Opening TIFF with memmap...")
# memmap keeps the array on disk
scan_memmap = tifffile.memmap(tif_file)
print(f"   Full Volume Shape: {scan_memmap.shape}, dtype: {scan_memmap.dtype}")

# Downsample by 4x across all axes to reduce memory by 64x
print("2. Downsampling scan 4x for safety...")
scan_small = scan_memmap[::4, ::4, ::4].copy()
print(f"   Downsampled Shape: {scan_small.shape}, RAM footprint: {scan_small.nbytes / 1e6:.2f} MB")

print("3. Voxelizing STL at matching low resolution...")
mesh = trimesh.load(stl_file)
# Voxelize with pitch scaled to match downsampled grid
pitch = 1.0  # Adjust pitch to approximate voxel spacing
cad_voxels = mesh.voxelized(pitch=pitch).matrix

print("Pipeline initialized successfully without OOM!")