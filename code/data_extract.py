import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
import random

# Path to KiTS23 dataset root
data_dir = r'kits23/dataset'
# List all case folders and take 200 randomly
random.seed(97)
start_index = random.randint(0, 200)
case_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith('case_')])[start_index:start_index+200]

# Container for 2D segmentation slices
slices = []


for case in case_dirs:
    seg_path = os.path.join(data_dir, case, 'segmentation.nii.gz')
    # Load the 3D segmentation mask
    nii = nib.load(seg_path)
    seg_vol = nii.get_fdata().astype(np.uint8)  # NumPy array
    print(f"{case} done")
    print(f"{case}: volume shape = {seg_vol.shape}")

    # Extract the axial slice with the largest segmented area
    areas = [(seg_vol[:, :, i] > 0).sum() for i in range(seg_vol.shape[2])]
    best_slice_idx = int(np.argmax(areas))
    seg_slice = seg_vol[:, :, best_slice_idx]
    #print(f" -> best slice index = {best_slice_idx}, slice shape = {seg_slice.shape}")

    # Resize to 64x64
    seg_slice_resized = resize(seg_slice, (64, 64), order=0, preserve_range=True, anti_aliasing=False)
    seg_slice_resized = (seg_slice_resized > 0).astype(np.uint8)

    slices.append(seg_slice_resized)

# At this point, `slices` is a list of 50 binary 2D NumPy arrays (64x64)
# Stack into a data matrix for PCA/LLE etc.
X = np.array([s.flatten() for s in slices])  # shape (200, 4096)
print("Data matrix X shape:", X.shape)

np.save('X.npy', X) 

"""
Load
X = np.load('X.npy')
print(X.shape)  # → (50, 4096)
"""