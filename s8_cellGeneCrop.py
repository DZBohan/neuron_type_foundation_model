import argparse
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--report_csv", default=None)
    p.add_argument("--cells", type=int, default=200)
    p.add_argument("--genes", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--strata", type=int, default=4)
    p.add_argument("--min_gene_detect_frac", type=float, default=0.05)
    p.add_argument("--alpha_detect", type=float, default=1.0)
    p.add_argument("--beta_std", type=float, default=0.5)
    p.add_argument("--gene_dedupe_mode", choices=["make_unique", "collapse"], default="make_unique")
    p.add_argument("--cell_dedupe_mode", choices=["make_unique", "drop"], default="make_unique")
    p.add_argument("--drop_allzero_genes", action="store_true")
    p.add_argument("--drop_allzero_cells", action="store_true")
    return p.parse_args()

def _get_basis_matrix(adata):
    if hasattr(adata, "layers") and ("counts" in adata.layers):
        M = adata.layers["counts"]
        basis = "layers['counts']"
    else:
        M = adata.X
        basis = "X"
    return M, basis

def make_names_unique_or_drop(adata, gene_mode="make_unique", cell_mode="make_unique"):
    if cell_mode == "make_unique":
        adata.obs_names_make_unique()
    elif cell_mode == "drop":
        obs = adata.obs.copy()
        dup_mask = obs.index.duplicated(keep="first")
        if dup_mask.sum() > 0:
            adata._inplace_subset_obs(~dup_mask)
    else:
        raise ValueError
    if gene_mode == "make_unique":
        adata.var_names_make_unique()
    elif gene_mode == "collapse":
        names = adata.var_names.to_numpy()
        uniq_names, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
        if counts.max() == 1:
            return
        M_basis, basis_name = _get_basis_matrix(adata)
        is_sparse = sp.issparse(M_basis)
        M_basis = M_basis.tocsr() if is_sparse else np.asarray(M_basis)
        n_cells = adata.n_obs
        n_unique = uniq_names.size
        def _collapse_matrix(M):
            is_sp = sp.issparse(M)
            M = M.tocsr() if is_sp else np.asarray(M)
            if is_sp:
                out = sp.csr_matrix((n_cells, n_unique), dtype=M.dtype)
            else:
                out = np.zeros((n_cells, n_unique), dtype=M.dtype)
            for j_old, j_new in enumerate(inverse):
                if is_sp:
                    out[:, j_new] = out[:, j_new] + M[:, j_old]
                else:
                    out[:, j_new] += M[:, j_old]
            return out
        new_basis = _collapse_matrix(M_basis)
        if basis_name == "X":
            adata.X = new_basis
        else:
            adata.layers["counts"] = new_basis
        if basis_name != "X":
            adata.X = _collapse_matrix(adata.X)
        if hasattr(adata, "layers"):
            for key in list(adata.layers.keys()):
                if key == "counts":
                    continue
        new_var = adata.var.groupby(adata.var_names).first().reindex(uniq_names)
        adata.var = new_var
        adata.var_names = uniq_names
    else:
        raise ValueError

def drop_all_zero_rows_cols(adata, drop_genes=True, drop_cells=True):
    M, basis_name = _get_basis_matrix(adata)
    gene_nz = np.asarray((M > 0).sum(axis=0)).ravel()
    cell_nz = np.asarray((M > 0).sum(axis=1)).ravel()
    if drop_genes:
        keep_g = gene_nz > 0
        if (~keep_g).sum() > 0:
            adata._inplace_subset_var(keep_g)
    if drop_cells:
        keep_c = cell_nz > 0
        if (~keep_c).sum() > 0:
            adata._inplace_subset_obs(keep_c)

def summarize_sparsity(X):
    nnz = int((X > 0).sum())
    total = int(X.size)
    frac_nz = nnz / total if total > 0 else 0.0
    per_cell_nz = (X > 0).sum(axis=1)
    per_gene_nz = (X > 0).sum(axis=0)
    return {
        "shape": [int(X.shape[0]), int(X.shape[1])],
        "nonzero_fraction": float(round(frac_nz, 6)),
        "median_nonzero_per_cell": float(np.median(per_cell_nz)),
        "median_nonzero_per_gene": float(np.median(per_gene_nz)),
    }

def select_hello_world_slice(
    X,
    target_cells=200,
    target_genes=500,
    seed=42,
    n_cell_strata=4,
    min_gene_detect_frac=0.05,
    alpha_detect=1.0,
    beta_std=0.5,
):
    rng = np.random.RandomState(seed)
    n_cells, n_genes_total = X.shape
    target_cells = min(target_cells, n_cells)
    cell_nonzero = (X > 0).sum(axis=1)
    quantiles = np.linspace(0, 1, n_cell_strata + 1)
    q_vals = np.unique(np.quantile(cell_nonzero, quantiles))
    bins = np.digitize(cell_nonzero, q_vals[1:-1], right=True) if q_vals.size > 2 else np.zeros_like(cell_nonzero)
    selected_cells = []
    n_strata = max(1, (q_vals.size - 1))
    per_stratum = int(np.ceil(target_cells / n_strata))
    for s in range(n_strata):
        idx_in_s = np.where(bins == s)[0]
        if idx_in_s.size == 0:
            continue
        take = min(per_stratum, idx_in_s.size)
        choice = rng.choice(idx_in_s, size=take, replace=False)
        selected_cells.append(choice)
    selected_cells = np.concatenate(selected_cells, axis=0)
    if selected_cells.size > target_cells:
        selected_cells = rng.choice(selected_cells, size=target_cells, replace=False)
    X_sub_cells = X[selected_cells, :]
    gene_detect_frac = (X_sub_cells > 0).mean(axis=0)
    gene_std = X_sub_cells.std(axis=0)
    keep_mask = gene_detect_frac >= min_gene_detect_frac
    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size < target_genes:
        sorted_genes = np.argsort(gene_detect_frac)[::-1]
        keep_idx = sorted_genes[:max(target_genes, min(2000, n_genes_total))]
    eps = 1e-8
    det_w = np.power(gene_detect_frac[keep_idx] + eps, alpha_detect)
    std_w = np.power(gene_std[keep_idx] + eps, beta_std)
    weights = det_w * std_w
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    take_genes = min(target_genes, keep_idx.size)
    selected_genes = rng.choice(keep_idx, size=take_genes, replace=False, p=weights)
    selected_cells.sort()
    selected_genes.sort()
    return selected_cells, selected_genes

def main():
    args = parse_args()
    sc.settings.verbosity = 1
    if not os.path.exists(args.input):
        sys.exit(1)
    adata = sc.read_h5ad(args.input)
    make_names_unique_or_drop(
        adata,
        gene_mode=args.gene_dedupe_mode,
        cell_mode=args.cell_dedupe_mode,
    )
    if args.drop_allzero_genes or args.drop_allzero_cells:
        drop_all_zero_rows_cols(
            adata,
            drop_genes=args.drop_allzero_genes,
            drop_cells=args.drop_allzero_cells,
        )
    if hasattr(adata, "layers") and ("counts" in adata.layers):
        X_for_sampling = adata.layers["counts"]
        X_for_sampling = X_for_sampling.toarray() if sp.issparse(X_for_sampling) else X_for_sampling
        input_basis = "layers['counts']"
    else:
        X_for_sampling = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        input_basis = "X"
    before_stats = summarize_sparsity(X_for_sampling)
    sel_cells, sel_genes = select_hello_world_slice(
        X_for_sampling,
        target_cells=args.cells,
        target_genes=args.genes,
        seed=args.seed,
        n_cell_strata=args.strata,
        min_gene_detect_frac=args.min_gene_detect_frac,
        alpha_detect=args.alpha_detect,
        beta_std=args.beta_std,
    )
    adata_crop = adata[sel_cells, sel_genes].copy()
    if hasattr(adata_crop, "layers") and ("counts" in adata_crop.layers):
        X_after = adata_crop.layers["counts"]
        X_after = X_after.toarray() if sp.issparse(X_after) else X_after
    else:
        X_after = adata_crop.X.toarray() if sp.issparse(adata_crop.X) else adata_crop.X
    after_stats = summarize_sparsity(X_after)
    adata_crop.uns = adata_crop.uns.copy() if adata_crop.uns is not None else {}
    adata_crop.uns["crop_params"] = {
        "input_file": os.path.abspath(args.input),
        "input_basis": input_basis,
        "cells": args.cells,
        "genes": args.genes,
        "seed": args.seed,
        "strata": args.strata,
        "min_gene_detect_frac": args.min_gene_detect_frac,
        "alpha_detect": args.alpha_detect,
        "beta_std": args.beta_std,
        "gene_dedupe_mode": args.gene_dedupe_mode,
        "cell_dedupe_mode": args.cell_dedupe_mode,
        "drop_allzero_genes": bool(args.drop_allzero_genes),
        "drop_allzero_cells": bool(args.drop_allzero_cells),
    }
    adata_crop.uns["sparsity_summary"] = {"before": before_stats, "after": after_stats}
    adata_crop.uns["crop_indices"] = {
        "selected_cell_indices_from_input": sel_cells.tolist(),
        "selected_gene_indices_from_input": sel_genes.tolist(),
    }
    if args.report_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_csv)), exist_ok=True)
        df = pd.DataFrame([
            {"stage": "before", **before_stats},
            {"stage": "after",  **after_stats},
        ])
        df.to_csv(args.report_csv, index=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    adata_crop.write_h5ad(args.output, compression="gzip")

if __name__ == "__main__":
    main()