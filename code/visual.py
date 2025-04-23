import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.transform import resize

def show_volumne(seg):
    print("Segmentation volume shape:", seg.shape)
    label = 1  # Change to 2 for tumor

    verts, faces, _, _ = measure.marching_cubes(seg == label, level=0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.6)
    mesh.set_facecolor('green' if label == 1 else 'red')

    ax.add_collection3d(mesh)
    ax.set_xlim(0, seg.shape[0])
    ax.set_ylim(0, seg.shape[1])
    ax.set_zlim(0, seg.shape[2])

    ax.set_title('3D Visualization of Kidney' if label == 1 else 'Tumor')
    plt.tight_layout()
    plt.show()

def show_slices(seg):
    non_zero_slices = [i for i in range(seg.shape[2]) if np.any(seg[:, :, i])]
    print("Non-zero slices = ", len(non_zero_slices))
    # Pick 12 from the middle of the non-zero region
    mid = len(non_zero_slices) // 2
    selected = non_zero_slices[mid - 4: mid + 5]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flatten()

    for ax, idx in zip(axes, selected):
        kidney = (seg[:, :, idx] == 1)
        tumor = (seg[:, :, idx] == 2)

        # Create RGB image
        color_mask = np.zeros((*kidney.shape, 3), dtype=np.uint8)
        color_mask[kidney] = [0, 255, 0]   # Green for kidney
        color_mask[tumor] = [255, 0, 0]    # Red for tumor

        ax.imshow(color_mask)
        ax.set_title(f"Slice {idx}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_slices(slice):
    assert slice.shape[0] == 64 and slice.shape[1] == 64, f"Shape error, wanted 64x64 but got {slice.shape}"

    plt.imshow(slice, cmap="gray")
    plt.show()

if __name__=="__main__":

    index = input("Enter an index value between 0 to 489.\n")

    # Example: Load segmentation volume from first case
    seg_path = rf'kits23/dataset/case_{index.zfill(5)}/segmentation.nii.gz'
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)

    show_volumne(seg)
    show_slices(seg)

    areas = [(seg[:, :, i] > 0).sum() for i in range(seg.shape[2])]
    best_slice_idx = int(np.argmax(areas))
    seg_slice = seg[:, :, best_slice_idx]

    seg_slice_resized = resize(seg_slice, (64, 64), order=0, preserve_range=True, anti_aliasing=False)
    seg_slice_resized = (seg_slice_resized > 0).astype(np.uint8)

    plot_slices(seg_slice_resized)

    """
    197 - tumour
    """
