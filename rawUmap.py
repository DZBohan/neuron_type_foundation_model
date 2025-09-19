#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot cortical embedding using provided x,y from cell_metadata.csv.
- Colors = Top-K base clusters (by cell count), others -> "Others"
- Base cluster is parsed from cluster_annotation_term.csv["description"]
  (e.g., "MGE interneuron (subcluster 7)" -> "MGE interneuron")
- Highlight any number of subclusters by cluster_alias (optional)
- Legend is always placed OUTSIDE (right side), never blocking the plot.
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- utils ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-meta", required=True,
                   help="Path to cell_metadata.csv (must contain columns: x, y, cluster_alias)")
    p.add_argument("--cluster", required=True,
                   help="Path to cluster.csv (must contain columns: cluster_alias, label)")
    p.add_argument("--annot", required=True,
                   help="Path to cluster_annotation_term.csv (must contain columns: label, description)")
    p.add_argument("--out", default="cortical_umap.png", help="Output figure path (png/pdf/svg)")
    p.add_argument("--title", default="Cortical UMAP – Top-K clusters", help="Figure title")
    p.add_argument("--top-k", type=int, default=10, help="How many top clusters to color explicitly")
    p.add_argument("--highlight-subclusters", default="",
                   help="Comma-separated cluster_alias id list to highlight (e.g. '2523,210'). "
                        "Empty means no highlight.")
    return p.parse_args()

def strip_parentheses(desc: str) -> str:
    """Return text before the first ' ('  (“Base cluster name”)."""
    if not isinstance(desc, str):
        return "Unknown"
    m = re.split(r"\s*\(", desc.strip(), maxsplit=1)
    return m[0].strip() if m else desc.strip()

def build_cluster_alias_to_base(cluster_csv, annot_csv):
    """Map cluster_alias -> base_cluster_name."""
    # cluster.csv: columns typically include ['cluster_alias','number_of_cells','label']
    cl = pd.read_csv(cluster_csv)
    req1 = {"cluster_alias", "label"}
    if not req1.issubset(set(cl.columns)):
        raise ValueError(f"[cluster.csv] missing columns: {sorted(list(req1 - set(cl.columns)))}")
    cl = cl[["cluster_alias", "label"]].dropna()

    # cluster_annotation_term.csv: columns include ['label','description', ...]
    an = pd.read_csv(annot_csv)
    req2 = {"label", "description"}
    if not req2.issubset(set(an.columns)):
        raise ValueError(f"[cluster_annotation_term.csv] missing columns: {sorted(list(req2 - set(an.columns)))}")
    an = an[["label", "description"]].dropna()

    merged = cl.merge(an, on="label", how="left")
    merged["base_cluster"] = merged["description"].apply(strip_parentheses).fillna("Unknown")
    # Return Series: index = cluster_alias (int), value = base_cluster (str)
    # Ensure cluster_alias is int if possible:
    try:
        merged["cluster_alias"] = merged["cluster_alias"].astype(int)
    except Exception:
        pass
    return merged.set_index("cluster_alias")["base_cluster"]

def make_palette(names):
    """Return a dict name->color using tab20 + fallback."""
    base_cmap = plt.get_cmap("tab20")
    colors = [base_cmap(i % 20) for i in range(len(names))]
    return {n: c for n, c in zip(names, colors)}

# ---------- main ----------

def main():
    args = parse_args()

    # Read metadata (must contain x,y,cluster_alias)
    meta = pd.read_csv(args.cell_meta)
    need_cols = {"x", "y", "cluster_alias"}
    if not need_cols.issubset(set(meta.columns)):
        raise ValueError(f"[cell_metadata.csv] missing columns: {sorted(list(need_cols - set(meta.columns)))}")

    # Normalize cluster_alias dtype
    try:
        meta["cluster_alias"] = meta["cluster_alias"].astype(int)
    except Exception:
        pass

    # Map cluster_alias -> base cluster name
    alias2base = build_cluster_alias_to_base(args.cluster, args.annot)

    # Attach base cluster to each cell
    meta["base_cluster"] = meta["cluster_alias"].map(alias2base).fillna("Unknown")

    # Top-K selection by cell count
    counts = meta["base_cluster"].value_counts()
    top_names = counts.index[: args.top_k].tolist()
    meta["plot_cluster"] = np.where(meta["base_cluster"].isin(top_names),
                                    meta["base_cluster"], "Others")

    # Colors: Top-K distinct + "Others"=gray
    palette = make_palette(top_names)
    palette["Others"] = (0.7, 0.7, 0.7, 0.6)

    # Prepare highlight list (any number allowed)
    highlight_raw = [s for s in args.highlight_subclusters.split(",") if s.strip() != ""]
    try:
        highlight_ids = set(int(s) for s in highlight_raw)
    except Exception:
        highlight_ids = set()  # ignore malformed
    meta["is_highlight"] = meta["cluster_alias"].isin(highlight_ids)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # 1) draw background clusters (Top-K + Others)
    for name in top_names + ["Others"]:
        df = meta[meta["plot_cluster"] == name]
        if df.empty: 
            continue
        ax.scatter(df["x"], df["y"], s=0.01, alpha=0.8, c=[palette[name]], label=name, linewidths=0)

    # 2) overlay highlighted subclusters as hollow circles
    if len(highlight_ids) > 0:
        dfh = meta[meta["is_highlight"]]
        ax.scatter(dfh["x"], dfh["y"], s=0.01, facecolors="none", edgecolors="black",
                   linewidths=0.6, alpha=0.9, label="Highlighted subclusters")

    ax.set_title(args.title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Legend OUTSIDE on the right, always
    lgd = ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        markerscale=70.0
    )

    plt.tight_layout()
    # bbox_inches="tight" to include outside legend
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[done] saved: {args.out}")

if __name__ == "__main__":
    main()

