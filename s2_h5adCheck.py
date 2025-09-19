# 有多少细胞和基因

# obs 里存放了哪些细胞信息（是否有 cell_type）

# var 里有哪些基因相关的注释

# uns 里存放了哪些未结构化数据

import scanpy as sc

adata = sc.read_h5ad("/coh_labs/dits/bzhang/test/hWorldFM2/subset_2523_210_Cerebral-cortex.screened.h5ad")

print("=== 基本信息 ===")
print(f"Cells (obs): {adata.n_obs}")
print(f"Genes (var): {adata.n_vars}")

print("\n=== obs (细胞 metadata 列) ===")
print(adata.obs.columns.tolist())

if "cell_type" in adata.obs.columns:
    print("\n前 10 个细胞的 cell_type 标签：")
    print(adata.obs["cell_type"].head(10))
else:
    print("\n 没有找到 cell_type 列")

print("\n=== var (基因 metadata 列) ===")
print(adata.var.columns.tolist())

print("\n=== uns (其他未结构化 metadata 键) ===")
print(list(adata.uns.keys()))

# 打印 anatomical_division_label 这一列的基本信息
print("=== anatomical_division_label (summary) ===")
print("Unique labels:", adata.obs["anatomical_division_label"].unique().tolist())
print("\nTop 10 categories with counts:")
print(adata.obs["anatomical_division_label"].value_counts().head(10))

# 打印前 10 个 example 值
print("\nExample values:")
print(adata.obs["anatomical_division_label"].head(10).tolist())
