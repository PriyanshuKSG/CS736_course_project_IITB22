import os
import numpy as np
from PIL import Image

X = []
root_dir = r'data_used'  # ← replace this with your actual path

for dirpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.lower().endswith('.gif'):
            fpath = os.path.join(dirpath, fname)
            try:
                img = Image.open(fpath).convert('L')  # use only the first frame if animated
                img = img.resize((64, 64))
                img_arr = np.array(img).flatten()
                X.append(img_arr)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")

X = np.array(X)
np.save('mpeg.npy', X)
print("Saved mpeg.npy with shape:", X.shape)
