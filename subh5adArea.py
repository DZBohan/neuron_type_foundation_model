# filter_by_region
#
# Create a new h5ad containing only cells from a chosen region.
# Example:
#   python filter_by_region.py \
#       --in subset_2523_210.h5ad \
#       --out subset_2523_210_cortex.h5ad \
#       --region "Cerebral cortex"
#
# You can also change the region column if needed:
#   --region-col anatomical_division_label

import argparse
import anndata as ad
import pandas as pd  # optional, nicer summaries

def main():
    ap = argparse.ArgumentParser(description="Filter an h5ad by region label.")
    ap.add_argument("--in",  dest="in_path",  required=True, help="Input .h5ad")
    ap.add_argument("--out", dest="out_path", required=True, help="Output .h5ad")
    ap.add_argument("--region", required=True, help="Region name to keep (exact match)")
    ap.add_argument("--region-col", default="anatomical_division_label",
                    help="obs column that holds region labels (default: anatomical_division_label)")
    args = ap.parse_args()

    print("[info] loading AnnData:", args.in_path)
    adata = ad.read_h5ad(args.in_path)   # file is small enough to load in memory

    if args.region_col not in adata.obs.columns:
        raise SystemExit(f"[error] obs does not have column '{args.region_col}'. "
                         f"Available: {list(adata.obs.columns)}")

    # Build boolean mask and subset
    mask = adata.obs[args.region_col] == args.region
    n_keep = int(mask.sum())
    n_total = adata.n_obs
    if n_keep == 0:
        raise SystemExit(f"[error] No cells matched region='{args.region}' "
                         f"in column '{args.region_col}'.")

    print(f"[info] filtering: keep region == '{args.region}' "
          f"({n_keep}/{n_total} cells)")

    adata_sub = adata[mask].copy()

    # Optional: brief sanity print of the cluster label column if present
    for cand in ("subtype", "cluster", "cluster_alias", "cluster_label", "label"):
        if cand in adata_sub.obs.columns:
            vc = adata_sub.obs[cand].astype(str).value_counts().head(10)
            print(f"[info] top categories in '{cand}' (first 10):\n{vc}")
            break

    print("[info] writing:", args.out_path)
    # Use compression to keep it compact; change to None if you need raw speed
    adata_sub.write_h5ad(args.out_path, compression="gzip")
    print("[done]")

if __name__ == "__main__":
    main()
