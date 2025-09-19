# Check if cell_barcode in metadata matches obs_names in h5ad

import anndata as ad
import pandas as pd

adata = ad.read_h5ad("WHB-10Xv3-Neurons-raw.h5ad")
meta = pd.read_csv("cell_metadata.csv")

# Quick preview
print("obs_names example:", adata.obs_names[:5].tolist())
print("cell_label example:", meta["cell_label"].head().tolist())
print("cell_barcode example:", meta["cell_barcode"].head().tolist())