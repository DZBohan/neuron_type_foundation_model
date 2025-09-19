import scanpy as sc

adata = sc.read_h5ad("/coh_labs/dits/bzhang/test/hWorldFM2/subset_2523_210_Cerebral-cortex.screened.h5ad")

print("=== Basic Info ===")
print(f"Cells (obs): {adata.n_obs}")
print(f"Genes (var): {adata.n_vars}")

print("\n=== obs (cell metadata columns) ===")
print(adata.obs.columns.tolist())

if "cell_type" in adata.obs.columns:
    print("\nFirst 10 cell_type labels:")
    print(adata.obs["cell_type"].head(10))
else:
    print("\nNo cell_type column found")

print("\n=== var (gene metadata columns) ===")
print(adata.var.columns.tolist())

print("\n=== uns (other unstructured metadata keys) ===")
print(list(adata.uns.keys()))

# Print summary of anatomical_division_label column
print("=== anatomical_division_label (summary) ===")
print("Unique labels:", adata.obs["anatomical_division_label"].unique().tolist())
print("\nTop 10 categories with counts:")
print(adata.obs["anatomical_division_label"].value_counts().head(10))

# Print first 10 example values
print("\nExample values:")
print(adata.obs["anatomical_division_label"].head(10).tolist())
