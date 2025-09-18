import argparse
import anndata as ad
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Subset an h5ad by obs_names using metadata (fast path, no join)."
    )
    p.add_argument("--adata", required=True, help="Input .h5ad")
    p.add_argument("--meta", required=True, help="Cell metadata .csv")
    p.add_argument(
        "--obsname-col", default="cell_label",
        help="Column in metadata that exactly matches adata.obs_names (default: cell_label)"
    )
    p.add_argument(
        "--label-col", default="cluster_alias",
        help="Metadata column that contains cluster IDs to keep (default: cluster_alias)"
    )
    p.add_argument(
        "--ids", required=True,
        help="Comma-separated cluster IDs to keep, e.g. 2523,210"
    )
    p.add_argument("--out", required=True, help="Output .h5ad")
    return p.parse_args()

def main():
    args = parse_args()
    keep_ids = [s.strip() for s in args.ids.split(",") if s.strip()]

    print("[info] loading metadata CSV ...")
    usecols = [args.obsname_col, args.label_col]
    meta = pd.read_csv(args.meta, usecols=usecols, dtype={args.obsname_col: str})
    # 保守做法：把 label 列转成字符串再比对
    meta[args.label_col] = meta[args.label_col].astype(str)

    meta_sel = meta[meta[args.label_col].isin(keep_ids)].copy()
    if meta_sel.empty:
        raise SystemExit(
            f"[error] No rows in metadata where {args.label_col} in {keep_ids}."
        )

    # 给每个 cluster 生成 subtype 名称
    id2sub = {cid: f"cluster_{cid}" for cid in keep_ids}
    meta_sel["subtype"] = meta_sel[args.label_col].map(id2sub)

    print("[info] loading AnnData (backed) ...")
    adata = ad.read_h5ad(args.adata, backed="r")

    # 直接用 obs_names 与 metadata 的 obsname 列取交集
    obs_idx = pd.Index(adata.obs_names.astype(str))
    wanted = pd.Index(meta_sel[args.obsname_col].astype(str))
    keep_obs = obs_idx.intersection(wanted)

    if keep_obs.empty:
        raise SystemExit(
            "[error] No overlap between adata.obs_names and "
            f"metadata[{args.obsname_col}]."
        )

    print(f"[info] cells in clusters {keep_ids}: {len(keep_obs)}")

    # 只把需要的行读入内存
    sub = adata[keep_obs, :].to_memory()

    # 按 obs_name 对齐写入 subtype
    meta_map = meta_sel.set_index(args.obsname_col)["subtype"]
    sub.obs["subtype"] = pd.Series(index=sub.obs_names, dtype="object")
    sub.obs.loc[meta_map.index.intersection(sub.obs.index), "subtype"] = \
        meta_map.loc[meta_map.index.intersection(sub.obs.index)].values

    # 简单 sanity check
    print("[info] subtype counts:")
    print(sub.obs["subtype"].value_counts(dropna=False))

    print(f"[info] writing {args.out} ...")
    sub.write_h5ad(args.out)
    print("[done]")

if __name__ == "__main__":
    main()

