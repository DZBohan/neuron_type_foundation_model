# subh5adCheck.py
import os
import anndata as ad
import pandas as pd  # 仅用于更友好的输出（可选）

PATH = "/coh_labs/dits/bzhang/test/hWorldFM2/subset_2523_210.h5ad"

def human_size(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024 or u == "TB":
            return f"{n:.2f} {u}"
        n /= 1024

# 1) 读取 (backed 读取更省内存, 只够访问 obs)
adata = ad.read_h5ad(PATH, backed="r")

# 2) 自动识别 "cluster/subtype" 列
obs_cols = list(adata.obs.columns)
candidate_cols = ["cluster", "subtype", "cluster_alias", "cluster_label", "label"]
cluster_col = next((c for c in candidate_cols if c in obs_cols), None)
if cluster_col is None:
    raise SystemExit(f"[error] No cluster-like column found. obs columns: {obs_cols}")

# 3) 全局统计
s = adata.obs[cluster_col].astype(str)
counts = s.value_counts()
n_2523 = counts.get("cluster_2523", 0) + counts.get("2523", 0)
n_210 = counts.get("cluster_210", 0) + counts.get("210", 0)

print("=== Summary ===")
print(f"File: {PATH}")
print(f"Total cells: {adata.n_obs}")
print(f"Cells in cluster 2523: {int(n_2523)}")
print(f"Cells in cluster 210 : {int(n_210)}")

print("\n=== obs column used ===")
print(f"Column name: {cluster_col}")
print("Example values:", s.head(10).tolist())

# 3b) Cerebral cortex 内部检查
if "anatomical_division_label" in adata.obs.columns:
    cortex = adata.obs[adata.obs["anatomical_division_label"] == "Cerebral cortex"]
    cortex_counts = cortex[cluster_col].astype(str).value_counts()
    c_2523 = cortex_counts.get("cluster_2523", 0) + cortex_counts.get("2523", 0)
    c_210 = cortex_counts.get("cluster_210", 0) + cortex_counts.get("210", 0)
    print("\n=== Cerebral cortex only ===")
    print(f"Cells in cortex total: {cortex.shape[0]}")
    print(f"  - cluster_2523: {c_2523}")
    print(f"  - cluster_210 : {c_210}")
else:
    print("\n[warn] anatomical_division_label column not found in obs!")

# 4) 文件大小
size_bytes = os.path.getsize(PATH)
print("\n=== File size ===")
print(human_size(size_bytes))
