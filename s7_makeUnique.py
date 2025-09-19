import argparse
import scanpy as sc
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Ensure unique gene and cell IDs in AnnData")
    parser.add_argument("--infile", required=True, help="Input .h5ad file")
    parser.add_argument("--outfile", required=True, help="Output .h5ad file with unique IDs")
    args = parser.parse_args()

    # Load dataset
    print(f"[info] Loading {args.infile} ...")
    adata = sc.read_h5ad(args.infile)

    # Make gene and cell names unique
    print("[info] Making var_names and obs_names unique (if needed) ...")
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Check duplicates
    dup_vars = pd.Index(adata.var_names).duplicated().sum()
    dup_obs  = pd.Index(adata.obs_names).duplicated().sum()
    print(f"[check] Duplicated gene names: {dup_vars}")
    print(f"[check] Duplicated cell IDs : {dup_obs}")

    # Save
    adata.write_h5ad(args.outfile)
    print(f"[done] Saved to {args.outfile}")

if __name__ == "__main__":
    main()
