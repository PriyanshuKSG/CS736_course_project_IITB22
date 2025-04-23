# 🧠 Shape Analysis using PCA, KPCA, and LLE

This project explores shape representation and retrieval using dimensionality reduction techniques — Principal Component Analysis (PCA), Kernel PCA (KPCA), and Locally Linear Embedding (LLE). We apply these methods to medical images from the KiTS23 dataset and generic shapes from the MPEG-7 dataset. The goal is to analyze shape variations, learn compact representations, and evaluate retrieval performance across and within datasets.

---

## 📁 Folder Structure
```bash
.
├── code/
│   ├── main.py             # Main driver script: applies PCA, KPCA, LLE, and performs shape retrieval
│   ├── visual.py           # Data visualization for better understanding of shapes
│   ├── data_extract.py     # Extracts largest kidney cross-section slice from KiTS23 CT volumes
│   ├── data_mpeg7.py       # Extracts shape descriptors from selected MPEG-7 classes
│
├── initial_data/           # Contains visualizations of raw input data (kidney and MPEG-7 shapes)
├── pca/                    # Contains visual results and embeddings from PCA
├── kpca/                   # Contains visual results and embeddings from Kernel PCA
├── lle/                    # Contains visual results and embeddings from LLE


---

## 📊 Datasets

### 1. MPEG-7 Shape Dataset  
- URL: [https://dabi.temple.edu/external/shape/MPEG7/dataset.html](https://dabi.temple.edu/external/shape/MPEG7/dataset.html)  
- A set of binary silhouette images representing various object classes.  
- We use a subset of classes, with 180 shape images resized to 64×64, resulting in a flattened feature shape of (180, 4096).

### 2. KiTS23 Kidney CT Dataset  
- URL: [https://github.com/neheller/kits23](https://github.com/neheller/kits23)  
- Contains 3D abdominal CT scans of patients along with kidney segmentations.  
- From each 3D volume, the slice with the largest kidney cross-section is extracted, resized to 64×64, and flattened to shape (200, 4096).

---

## 🔄 Combined Dataset for Cross-Domain Retrieval

To evaluate shape retrieval in a cross-domain setting:
- We combine **180 MPEG-7 shape images** with **20 randomly selected kidney slices**.
- The final dataset contains 200 images of size 4096 (flattened), used for retrieval experiments across different shape types.

---

## 🖼️ Output Visualizations

All visual results and retrieval experiments are saved in dedicated folders:

- `initial_data/` – Visualizations of original kidney and MPEG-7 shapes after preprocessing.
- `pca/` – Results and retrieval outputs using PCA.
- `kpca/` – Results and retrieval outputs using Kernel PCA.
- `lle/` – Results and retrieval outputs using LLE.

These include shape embeddings, nearest neighbor matches, and clustering patterns.

---

## Authors
This project was developed as part of the Medical Image Computing, CS736 course at IIT Bombay by 
Priyanshu Gangavati (22B2165) and Varad Patil (22B2270)

