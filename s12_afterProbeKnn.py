import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, confusion_matrix
)

import scanpy as sc

def plot_confusion_matrix(cm, labels=('Inhibitory (0)', 'Excitatory (1)'), out_png='cm.png', title='Confusion Matrix'):
    plt.figure(figsize=(4.2, 3.6))
    im = plt.imshow(cm, cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0,1], ['Pred 0','Pred 1'])
    plt.yticks([0,1], ['True 0','True 1'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def scatter_umap(adata, color, out_png, title, palette=None):
    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    if palette is None:
        sc.pl.umap(adata, color=color, ax=ax, show=False, size=8, alpha=0.8)
    else:
        sc.pl.umap(adata, color=color, ax=ax, show=False, size=8, alpha=0.8, palette=palette)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Unsupervised Louvain on probe embeddings with full visualization.")
    ap.add_argument("--emb", default="probe_outputs_v1/embeddings.npy", help="Embeddings .npy (cells × dims)")
    ap.add_argument("--lab", default="probe_outputs_v1/labels.npy", help="Binary labels .npy (0/1)")
    ap.add_argument("--outdir", default="unsup_louvain_outputs", help="Output directory")
    ap.add_argument("--n_neighbors", type=int, default=15, help="kNN neighbors (default: 15)")
    ap.add_argument("--metric", default="cosine", choices=["cosine","euclidean"], help="Distance metric for kNN")
    ap.add_argument("--resolution", type=float, default=0.8, help="Louvain resolution")
    ap.add_argument("--umap_min_dist", type=float, default=0.3, help="UMAP min_dist (vis only)")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed for UMAP")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load data
    X = np.load(args.emb)         # [N, D]
    y = np.load(args.lab).astype(int)  # [N], 0/1
    assert X.shape[0] == y.shape[0], "embeddings and labels must have same #cells"

    # 2) Z-score
    Xz = StandardScaler().fit_transform(X)

    # 3) Build AnnData + neighbors graph
    ad = sc.AnnData(X=Xz)
    sc.pp.neighbors(ad, n_neighbors=args.n_neighbors, metric=args.metric)

    # 4) UMAP for visualization only (does not affect clustering)
    sc.tl.umap(ad, min_dist=args.umap_min_dist, random_state=args.random_state)

    # 5) Louvain (fallback to Leiden if missing dependencies)
    used_method = "louvain"
    try:
        sc.tl.louvain(ad, resolution=args.resolution, key_added="louvain")
        pred = ad.obs["louvain"].astype("category").cat.codes.to_numpy()
    except Exception as e:
        print("[warn] Louvain failed (likely missing 'python-igraph' or 'louvain' package). Falling back to Leiden.")
        print("[warn] error:", repr(e))
        used_method = "leiden"
        sc.tl.leiden(ad, resolution=args.resolution, key_added="leiden")
        pred = ad.obs["leiden"].astype("category").cat.codes.to_numpy()

    # 6) Metrics vs. true labels (0/1)
    ari = adjusted_rand_score(y, pred)
    nmi = normalized_mutual_info_score(y, pred)
    h   = homogeneity_score(y, pred)
    c   = completeness_score(y, pred)

    # Generate a “best-aligned” binary prediction from true binary labels:
    # simple strategy: map each cluster to {0/1} by majority label within the cluster
    df = pd.DataFrame({"pred": pred, "y": y})
    map_tbl = df.groupby("pred")["y"].agg(lambda s: 1 if (s.mean() >= 0.5) else 0).to_dict()
    bin_pred = np.array([map_tbl[p] for p in pred], dtype=int)

    cm = confusion_matrix(y, bin_pred)
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(
        os.path.join(args.outdir, f"confusion_matrix_{used_method}.csv"), index=True
    )

    # 7) Save metrics
    meta = {
        "method": used_method,
        "n_neighbors": args.n_neighbors,
        "metric": args.metric,
        "resolution": args.resolution,
        "UMAP_min_dist": args.umap_min_dist,
        "random_state": args.random_state,
        "n_cells": int(X.shape[0]),
        "n_dims": int(X.shape[1]),
        "n_clusters": int(len(np.unique(pred))),
        "ARI": float(ari),
        "NMI": float(nmi),
        "Homogeneity": float(h),
        "Completeness": float(c),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[metrics]", meta)

    # 8) Visualizations
    # 8.1 UMAP colored by predicted clusters
    key = "louvain" if used_method == "louvain" else "leiden"
    ad.obs[key] = pd.Categorical(pred.astype(str))
    scatter_umap(ad, color=key, out_png=os.path.join(args.outdir, f"umap_{used_method}.png"),
                 title=f"UMAP colored by {used_method} clusters")

    # 8.2 UMAP colored by true labels (0/1)
    ad.obs["true_label"] = pd.Categorical(y.astype(str))
    scatter_umap(ad, color="true_label", out_png=os.path.join(args.outdir, "umap_true_label.png"),
                 title="UMAP colored by true labels (0/1)", palette=["#1f77b4", "#d62728"])

    # 8.3 Confusion matrix heatmap
    plot_confusion_matrix(cm, out_png=os.path.join(args.outdir, f"cm_{used_method}.png"),
                          title=f"Confusion Matrix ({used_method}→bin)")

    # 8.4 Cluster size barplot + purity (proportion of majority class in cluster)
    cl_sizes = df.groupby("pred").size().rename("size")
    purity = df.groupby("pred")["y"].apply(lambda s: max((s==0).mean(), (s==1).mean())).rename("purity")
    stat = pd.concat([cl_sizes, purity], axis=1).reset_index().rename(columns={"pred":"cluster"})
    stat.to_csv(os.path.join(args.outdir, f"cluster_stats_{used_method}.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(6.5,4.2), dpi=150)
    ax1.bar(stat["cluster"], stat["size"], alpha=0.7)
    ax1.set_xlabel("Cluster ID"); ax1.set_ylabel("Size")
    ax2 = ax1.twinx()
    ax2.plot(stat["cluster"], stat["purity"], marker="o", linestyle="--")
    ax2.set_ylabel("Purity")
    plt.title(f"Cluster sizes & purity ({used_method})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"cluster_size_purity_{used_method}.png"), dpi=180)
    plt.close()

    print(f"[done] outputs saved to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
