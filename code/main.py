import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding

def silhouette_and_scatter_plot(Z, method):
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=97).fit(Z[:, :2])      # pick 3 or 4 clusters
    labels = kmeans.labels_
    print(f'Silhouette score {method} = ', silhouette_score(Z, labels))

    plt.figure(figsize=(6,6))
    plt.scatter(Z[:,0], Z[:,1], s=30, alpha=0.7)
    plt.title(f'{method} Embeddings')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.grid(True)
    plt.show()

def compute_sdf(mask):
    pos = distance_transform_edt(mask)
    neg = distance_transform_edt(1 - mask)
    return pos - neg

def perform_pca(sdf):
    
    pca = PCA(n_components=sdf.shape[0]).fit(sdf)

    ev = pca.explained_variance_
    plt.figure(figsize=(6,4))
    plt.plot(ev, marker='o', label='Eigenvalues')
    plt.xlabel('PC index')
    plt.ylabel('Variance')
    plt.legend()
    plt.show()

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cumvar, 0.95) + 1
    print(f"Number of PCs for 95% variance: k = {k}")

    Z = pca.transform(sdf)[:, :k]

    silhouette_and_scatter_plot(Z, method="PCA")

    return Z

def perform_kernelPCA(sdf):

    kpca = KernelPCA(n_components=sdf.shape[0], kernel='rbf', gamma=1.0/4096) 
    Z_kpca = kpca.fit_transform(sdf)

    eigenvalues = np.var(Z_kpca, axis=0)

    plt.plot(eigenvalues, marker='o')
    plt.title("KPCA Eigen Spectrum")
    plt.xlabel("Component Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()

    variance_ratios = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(variance_ratios)

    k = np.argmax(cumulative_variance >= 0.90) + 1  # +1 because argmax returns index

    print(f"Number of components to retain 90% variance: {k}")

    # Project KPCA embeddings to k components
    Z_kpca = Z_kpca[:, :k]

    silhouette_and_scatter_plot(Z_kpca, method="KPCA")

    return Z_kpca


def perform_lle(sdf, n_neighbors = 10):

    n_components = 2   # For 2D embedding
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')

    # Fit and transform
    Z_lle = lle.fit_transform(sdf)

    eigenvalues = np.var(Z_lle, axis=0)

    plt.plot(eigenvalues, marker='o')
    plt.title("LLE Eigen Spectrum")
    plt.xlabel("Component Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()

    variance_ratios = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(variance_ratios)

    k = np.argmax(cumulative_variance >= 0.90) + 1  # +1 because argmax returns index

    print(f"Number of components to retain 90% variance: {k}")

    # Project LLE embeddings to k components
    Z_lle = Z_lle[:, :k]

    silhouette_and_scatter_plot(Z_lle, method="LLE")

    return Z_lle


def perform_retrieval_per_class(vectorization, q_idx, Z, method, k = 5):

    masks = vectorization.reshape(-1, 64, 64)

    dists = euclidean_distances(Z[q_idx:q_idx+1], Z)[0]
    nn = np.argsort(dists)[1:1+k]   # top-k (skip itself)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes_flat = axes.flatten()

    # Plot the query in the first cell
    axes_flat[0].imshow(masks[q_idx], cmap='gray')
    axes_flat[0].set_title('Query')
    axes_flat[0].axis('off')

    # Plot the k nearest neighbors
    for i, idx in enumerate(nn):
        axes_flat[i+1].imshow(masks[idx], cmap='gray')
        axes_flat[i+1].set_title(f'Rank {i+1}')
        axes_flat[i+1].axis('off')

    plt.suptitle(f'Shape Retrieval {method}')
    plt.tight_layout()
    plt.show()

def perform_retrieval(vectorization, Z, method, k = 5, num_classes = 10):
    random.seed(97)
    q_idx = np.random.randint(0, 20)

    for index in range(num_classes):
        perform_retrieval_per_class(vectorization, 20*index + q_idx, Z, method)

def get_sdf(A):
    masks = A
    masks = masks.reshape(-1, 64, 64)

    sdf_list = []
    for m in masks:
        sdf_list.append(compute_sdf(m))
    
    A_sdf = np.stack(sdf_list).reshape(200, -1)  # (200,4096)

    return A_sdf

if __name__=="__main__":

    random.seed(97)

    X = np.load(r"X.npy")
    print(f"X.shape = {X.shape}")

    indices = np.random.choice(X.shape[0], size=20, replace=False)
    subset = X[indices]

    Y = np.load(r"mpeg.npy")
    combined = np.vstack((subset, Y))
    print("combined.shape = ",combined.shape)
    

    #pca_embeddings = perform_pca(get_sdf(X))
    #perform_retrieval(X, pca_embeddings, "PCA")

    #pca_embeddings_combined = perform_pca(get_sdf(combined))
    #perform_retrieval(combined, pca_embeddings_combined, "PCA")

    #kernelPCA_embeddings = perform_kernelPCA(get_sdf(X))
    #perform_retrieval(X, kernelPCA_embeddings, "KPCA")

    #kernelPCA_embeddings_combined = perform_kernelPCA(get_sdf(combined))
    #perform_retrieval(combined, kernelPCA_embeddings_combined, "KPCA")
    # Gamma to be tweaked, for best results. Can change kernels as well.


    #lle_embeddings = perform_lle(get_sdf(X))
    #perform_retrieval(X, lle_embeddings, "LLE")

    #lle_embeddings_combined = perform_lle(get_sdf(combined))
    #perform_retrieval(combined, lle_embeddings_combined, "LLE")