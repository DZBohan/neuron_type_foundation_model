import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

# 1. Load data
embeddings = np.load("probe_outputs_v1/embeddings.npy")   # cells × dimensions
labels = np.load("probe_outputs_v1/labels.npy")           # 0/1 labels

# 2. t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
emb_tsne = tsne.fit_transform(embeddings)

plt.figure(figsize=(6,6))
plt.scatter(emb_tsne[:,0], emb_tsne[:,1], c=labels, cmap="coolwarm", s=8, alpha=0.7)
plt.title("t-SNE of Probe Embeddings")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")
plt.colorbar(label="0=Inhibitory, 1=Excitatory")
plt.savefig("probe_outputs_v1/tsne_probe.png", dpi=150)
plt.show()

# 3. UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
emb_umap = umap_model.fit_transform(embeddings)

plt.figure(figsize=(6,6))
plt.scatter(emb_umap[:,0], emb_umap[:,1], c=labels, cmap="coolwarm", s=8, alpha=0.7)
plt.title("UMAP of Probe Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.colorbar(label="0=Inhibitory, 1=Excitatory")
plt.savefig("probe_outputs_v1/umap_probe.png", dpi=150)
plt.show()