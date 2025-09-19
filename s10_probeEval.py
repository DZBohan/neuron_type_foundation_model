#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Model (与预训练一致，并额外提供 encode() 接口以导出 embedding)
# -----------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, n_genes, emb_dim=32, nhead=4, nlayers=2, dropout=0.1, mask_token_id=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.token_embed = nn.Embedding(vocab_size + 1, emb_dim)  # +1 for mask id
        self.gene_embed  = nn.Embedding(n_genes, emb_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        """返回 logits（用于自监督训练）"""
        bsz, seqlen = x.shape
        tok = self.token_embed(x)
        gene_idx = torch.arange(seqlen, device=x.device, dtype=torch.long)
        gene = self.gene_embed(gene_idx).unsqueeze(0).expand(bsz, -1, -1)
        emb = tok + gene
        hidden = self.transformer(emb)
        logits = self.output(hidden)
        return logits

    @torch.no_grad()
    def encode(self, x):
        """返回最后一层 hidden（不经过输出层）。"""
        bsz, seqlen = x.shape
        tok = self.token_embed(x)
        gene_idx = torch.arange(seqlen, device=x.device, dtype=torch.long)
        gene = self.gene_embed(gene_idx).unsqueeze(0).expand(bsz, -1, -1)
        emb = tok + gene
        hidden = self.transformer(emb)  # [B, S, D]
        return hidden  # 直接返回隐状态


# -----------------------------
# Binning（与预训练一致）
# -----------------------------
def compute_bin_edges_per_gene(X, num_bins):
    qs = np.linspace(0, 1, num_bins + 1)
    edges_list = []
    for g in range(X.shape[1]):
        e = np.unique(np.quantile(X[:, g], qs))
        if e.size < 2:
            mn, mx = X[:, g].min(), X[:, g].max()
            if mx <= mn:
                mx = mn + 1e-8
            e = np.array([mn, mx])
        edges_list.append(e)
    return edges_list

def apply_bin_edges(X, edges_list, num_bins):
    B = np.zeros_like(X, dtype=int)
    for g in range(X.shape[1]):
        idx = np.searchsorted(edges_list[g], X[:, g], side="right") - 1
        B[:, g] = np.clip(idx, 0, num_bins - 1)
    return B


# -----------------------------
# 可视化工具
# -----------------------------
def plot_pca_scatter(emb, y, out_png, title="Probe embeddings (PCA-2D)"):
    pca = PCA(n_components=2, random_state=42)
    emb2 = pca.fit_transform(emb)
    plt.figure(figsize=(7,6))
    colors = np.array(['#1f77b4', '#d62728'])  # blue for 0, red for 1
    labels = np.array(['Inhibitory', 'Excitatory'])
    for cls in [0, 1]:
        sel = (y == cls)
        plt.scatter(emb2[sel, 0], emb2[sel, 1], s=8, c=colors[cls], label=labels[cls], alpha=0.6)
    plt.legend(frameon=True)
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_roc(y_true, y_score, out_png, title="ROC (Logistic probe)"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--', alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True, help="预训练后用于评估的 h5ad（与训练同源，同一 token 化）")
    ap.add_argument("--checkpoint", required=True, help="hello_model_best.pt")
    ap.add_argument("--binmeta", required=True, help="binning_meta.npz（保存了 edges 与 gene_names）")
    ap.add_argument("--use_log1p", action="store_true", help="是否在 normalize_total 后再 log1p（需与预训练一致）")
    ap.add_argument("--num_bins", type=int, default=20, help="与预训练一致的分箱数")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mask_token_id", type=int, default=20)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--outdir", default="probe_outputs")
    ap.add_argument("--label_col", default="subtype", help="细胞标签列（默认读取 subset_2523_210.h5ad 的 'subtype'）")
    # 默认将 cluster_2523 视为 excitatory(1)，cluster_210 视为 inhibitory(0)；可自行覆盖
    ap.add_argument("--exc_labels", default="cluster_2523", help="逗号分隔，映射为 1 的取值")
    ap.add_argument("--inh_labels", default="cluster_210", help="逗号分隔，映射为 0 的取值")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cpu")  # 若有 GPU 可改 "cuda"
    print(f"[info] device: {device}")

    # ---- 1) 读取 AnnData 并按训练方式归一化 ----
    adata = sc.read_h5ad(args.adata)
    X_work = adata.layers["counts"] if ("layers" in dir(adata) and "counts" in adata.layers) else adata.X
    X_work = X_work.toarray() if hasattr(X_work, "toarray") else np.asarray(X_work)
    sc.pp.normalize_total(adata, target_sum=1e4)
    if args.use_log1p:
        sc.pp.log1p(adata)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    # ---- 2) 载入 tokenizer 元数据：基因顺序 + 分箱边界 ----
    meta = np.load(args.binmeta, allow_pickle=True)
    edges_saved = list(meta["edges"])
    gene_names_saved = meta["gene_names"].astype(str)
    nbins_saved = int(meta["num_bins"])
    if nbins_saved != args.num_bins:
        raise ValueError(f"num_bins mismatch: saved={nbins_saved} vs arg={args.num_bins}")

    # 保证基因顺序一致（用训练时的顺序重排当前数据）
    current_gene_names = np.array(adata.var_names.values, dtype=str)
    idx = pd.Index(current_gene_names).get_indexer(gene_names_saved)
    if (idx < 0).any():
        missing = gene_names_saved[idx < 0]
        raise ValueError(f"这些训练基因在当前数据中缺失：{len(missing)} 例如 {missing[:10]}")
    X = X[:, idx]  # 现在 X 的列顺序与训练完全一致

    # ---- 3) 按保存的分箱边界对当前数据分箱 ----
    binned = apply_bin_edges(X, edges_saved, args.num_bins).astype(np.int64)

    # ---- 4) 构造标签：subtype → 0/1 ----
    if args.label_col not in adata.obs.columns:
        raise ValueError(f"找不到标签列 '{args.label_col}'；可用列：{list(adata.obs.columns)}")
    raw_labels = adata.obs[args.label_col].astype(str).values

    exc_set = set([x.strip() for x in args.exc_labels.split(",") if x.strip() != ""])
    inh_set = set([x.strip() for x in args.inh_labels.split(",") if x.strip() != ""])

    y = np.full(raw_labels.shape, fill_value=-1, dtype=int)
    y[np.isin(raw_labels, list(inh_set))] = 0
    y[np.isin(raw_labels, list(exc_set))] = 1
    if (y == -1).any():
        unknown = np.unique(raw_labels[y == -1])
        raise ValueError(f"有未映射的标签：{unknown}. 请通过 --exc_labels / --inh_labels 指明映射。")

    # ---- 5) 载入预训练模型，导出 embedding ----
    n_cells, n_genes = binned.shape
    model = SimpleTransformer(
        vocab_size=args.num_bins, n_genes=n_genes,
        emb_dim=args.emb_dim, nhead=args.nhead, nlayers=args.nlayers,
        dropout=args.dropout, mask_token_id=args.mask_token_id
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = TensorDataset(torch.tensor(binned, dtype=torch.long))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    all_emb = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            hidden = model.encode(xb)  # [B,S,D]
            emb = hidden.mean(dim=1)   # mean-pool over genes -> [B,D]
            all_emb.append(emb.cpu().numpy())
    all_emb = np.concatenate(all_emb, axis=0)  # [N, D]

    # 保存中间产物
    np.save(os.path.join(args.outdir, "embeddings.npy"), all_emb)
    np.save(os.path.join(args.outdir, "labels.npy"), y)
    pd.DataFrame({"cell": adata.obs_names.values, "label": y}).to_csv(
        os.path.join(args.outdir, "cells.tsv"), sep="\t", index=False
    )

    # ---- 6) 线性探针：Logistic Regression ----
    Xtr, Xte, ytr, yte = train_test_split(
        all_emb, y, test_size=args.val_ratio, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xtr, ytr)

    y_pred = clf.predict(Xte)
    y_score = clf.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, y_pred)
    f1  = f1_score(yte, y_pred)
    try:
        auc_roc = roc_auc_score(yte, y_score)
    except Exception:
        auc_roc = float("nan")

    print("\n=== Probe Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc_roc:.4f}")
    print("\nClassification report:")
    print(classification_report(yte, y_pred, target_names=["Inhibitory(0)", "Excitatory(1)"]))

    cm = confusion_matrix(yte, y_pred)
    pd.DataFrame(cm, index=["true_inh", "true_exc"], columns=["pred_inh", "pred_exc"]).to_csv(
        os.path.join(args.outdir, "confusion_matrix.csv")
    )

    # ---- 7) 可视化 ----
    plot_pca_scatter(all_emb, y, os.path.join(args.outdir, "pca_probe.png"))
    plot_roc(yte, y_score, os.path.join(args.outdir, "roc_probe.png"))

    # 保存模型与探针
    import joblib
    joblib.dump(clf, os.path.join(args.outdir, "logreg_probe.pkl"))

    # 记录元信息
    meta_json = {
        "adata": os.path.abspath(args.adata),
        "checkpoint": os.path.abspath(args.checkpoint),
        "binmeta": os.path.abspath(args.binmeta),
        "num_bins": args.num_bins,
        "emb_dim": args.emb_dim,
        "nhead": args.nhead,
        "nlayers": args.nlayers,
        "val_ratio": args.val_ratio,
        "use_log1p": bool(args.use_log1p),
        "label_col": args.label_col,
        "exc_labels": list(exc_set),
        "inh_labels": list(inh_set),
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(auc_roc),
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
    }
    pd.Series(meta_json).to_json(os.path.join(args.outdir, "run_meta.json"), indent=2)
    print(f"\n[done] outputs saved to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
