# Single-cell RNAseq Neuron Type Foundation Model

## Introduction

This project develops a Neuron Type Foundation Model trained on single-cell RNA-seq data. The core idea is to build a general-purpose representation model for neurons, enabling both self-supervised pre-training and downstream biological tasks.

The workflow consists of two major stages:

1. Pre-training

* Input AnnData .h5ad object containing cell × gene expression matrix and metadata.

* Use Masked bin prediction strategy, where continuous gene expression values are discretized into bins (tokens). Random bins are masked, and a transformer encoder is trained to reconstruct the missing tokens.

* Use validation perplexity to monitor model fit during pre-training.

2. Downstream Evaluation

* Train logistic regression probe on the learned embeddings to distinguish Excitatory vs. Inhibitory neurons.

* Evaluate via accuracy, F1-score, ROC-AUC, confusion matrix, and visualization with PCA/t-SNE/UMAP.

* Demonstrate that the model captures biologically meaningful features beyond simple reconstruction.

This work serves as a Hello World prototype for building foundation models in neuroscience. The trained embeddings provide a basis for future tasks, including disease state prediction, gene regulatory network inference, and cross-brain-region analysis.

## Dependencies

python==3.9

torch==1.13.0

numpy==1.24.4

pandas==1.5.3

scipy==1.13.1

scanpy==1.9.1

transformers==4.20.1

datasets==2.3.2

matplotlib==3.6.3

scikit-learn==1.2.2

umap-learn==0.5.5

tensorboard==2.8.0

## Anndata

Anndata WHB-10Xv3-Neurons-raw.h5ad is from [Allen Brain Cell Atlas](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas).

![[fig1]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig1.png?raw=true)

Progect: Transcriptomic diversity of cell types in adult human brain

![[fig2]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig2.png?raw=true)

Dataset: [Human whole-brain transcriptomic cell type atlas (Kimberly Siletti)](https://alleninstitute.github.io/abc_atlas_access/descriptions/WHB_dataset.html)

Download URL: [https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/expression_matrices/WHB-10Xv3/20240330/WHB-10Xv3-Neurons-raw.h5ad](https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/expression_matrices/WHB-10Xv3/20240330/WHB-10Xv3-Neurons-raw.h5ad)

## Metadata

1. cluster_annotation_term.csv

This file provides the biological annotations for each cluster and subcluster.

* Columns include label (unique cluster ID, e.g., CS202210140_2523) and description (human-readable category, e.g., "MGE interneuron (subcluster 5)").

* It serves as the dictionary to interpret what each cluster/subcluster biologically represents.

* Download URL: [https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/metadata/WHB-taxonomy/20240330/cluster_annotation_term.csv](https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/metadata/WHB-taxonomy/20240330/cluster_annotation_term.csv)

2. cluster.csv

This file maps the label (cluster ID) to a cluster_alias (numeric code).

* Example: CS202210140_2523 → cluster_alias 2523.

* It ensures consistency between the annotation terms and the cell-level metadata.

* Download URL: [https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/metadata/WHB-taxonomy/20240330/cluster.csv](https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/metadata/WHB-taxonomy/20240330/cluster.csv)


3. cell_metadata.csv

This file contains per-cell metadata, including each cell’s cluster assignment and embedding coordinates (x, y).

* Example columns: cell_label, cluster_alias, x, y.

* It allows us to link each single cell to its cluster/subcluster and visualize embeddings such as UMAP/t-SNE.

* Download URL: [https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/metadata/WHB-10Xv3/20241115/cell_metadata.csv](https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/metadata/WHB-10Xv3/20241115/cell_metadata.csv)

## Clusters

For this initial study, I selected two neuronal subclusters from the cerebral cortex:

1. Subcluster 2523 (CS20221040_3017)

* Cortical layer 5 excitatory neurons

* 682 cells

2. Subcluster 210 (CS20221040_704)

* Medial ganglionic eminence (MGE) interneurons

* 678 cells

Rationale of Selection

* Represents two fundamental neuronal types: Excitatory vs. Inhibitory.

* Both are located in nearby regions of the deep cerebral cortex, ensuring biological relevance.

* Each subcluster has a sufficient number of cells to support a "hello world" style pre-training experiment.

* The two subclusters have balanced cell numbers, which avoids class imbalance in downstream evaluation.

![[fig4]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig3.png?raw=true)

## Program 1: s1_rawUmap.py

This script visualizes cortical single-cell clusters using pre-computed embeddings from cell_metadata.csv. It merges cluster definitions from cluster.csv and cluster_annotation_term.csv, groups cells by their base cluster type, and highlights selected subclusters if specified. The output is a UMAP-like scatter plot showing the spatial distribution of the top-K clusters, with smaller clusters grouped as “Others”.

1. Parameters

--cell-meta : Path to cell_metadata.csv. Must contain columns: x, y, cluster_alias (cell embedding coordinates and subcluster ID).

--cluster : Path to cluster.csv. Provides the mapping between cluster_alias and label.

--annot : Path to cluster_annotation_term.csv. Provides human-readable description for each cluster label.

--out : Output file path for the figure (png/pdf/svg). Default: cortical_umap.png.

--title : Title of the figure. Default: "Cortical UMAP – Top-K clusters".

--top-k : Number of most frequent base clusters to explicitly color. Default: 10.

--highlight-subclusters : Comma-separated list of cluster_alias IDs to highlight (e.g., "2523,210"). Empty string means no highlight.

2. Output

The script produces a scatter plot showing cell embeddings colored by their base cluster type.

* The Top-K clusters are shown in distinct colors.

* All other clusters are merged into an “Others” category (gray).

* Any specified subclusters are overlaid as hollow black circles.

## Program 2: s2_h5adCheck.py

This script inspects the contents of a single-cell .h5ad Anndata. It prints summary information about the dataset, including the number of cells and genes, available metadata columns in both observations (obs) and variables (var), as well as any unstructured metadata (uns). In addition, it provides a detailed summary of the column anatomical_division_label, including unique categories, counts, and example values. This step is designed for exploratory data inspection before downstream analysis.

1. Parameters

Path to .h5ad file (hard-coded in the script):

2. Outputs

The script prints information to the terminal:

* Basic dataset info: number of cells and genes.

* Metadata columns in obs: list of available per-cell annotations (e.g., cell_type, anatomical_division_label).

* Metadata columns in var: list of gene-level annotations.

* Keys in uns: any additional stored metadata.

* Summary of anatomical_division_label:

	* All unique categories.

	* Top-10 categories with their cell counts.

	* Example values from the first 10 cells.

![[fig12]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig12.png?raw=true)

## Program 3: s3_h5adMergeCheck.py

This script is designed to verify the consistency of cell identifiers between a single-cell expression dataset stored in an .h5ad file and a metadata table stored in a .csv file. 

By previewing cell barcodes and labels from both sources, I can quickly check whether the metadata aligns correctly with the AnnData object.

2. parameters

* Anndata: Its .obs_names field holds the unique identifiers (cell barcodes) for each cell.

* cell_metadata.csv: It must include at least cell_barcode: the unique identifier for each cell, expected to match the AnnData .obs_names; and cell_label: cell type or other categorical information for each cell.

3. Outputs

The script prints quick previews to the terminal for verification:

* obs_names example: the first 5 cell barcodes from the AnnData file.

* cell_label example: the first 5 labels from the metadata CSV.

* cell_barcode example: the first 5 barcodes from the metadata CSV.

These outputs allow me to check whether the cell_barcode column in the metadata matches the obs_names in the .h5ad file.

## Program 4: s4_h5adCluster.py

This script subsets a single-cell Anndata stored in an .h5ad file based on metadata information. 

It selects specific clusters (given by their IDs in the metadata) and creates a smaller AnnData object containing only the cells from those clusters. 

Each selected cluster is annotated with a subtype label, and the subset is saved as a new .h5ad file.

1. Parameters

* --adata (.h5ad): The input AnnData file containing the single-cell expression matrix.

* --meta (.csv): Metadata table providing cell annotations. Must contain at least:

* obsname-col (default: cell_label): matches adata.obs_names for alignment.

* label-col (default: cluster_alias): specifies cluster IDs.

* --obsname-col (optional, default=cell_label): Column in the metadata that exactly matches the cell identifiers in the .h5ad file.

* --label-col (optional, default=cluster_alias): Column in the metadata containing cluster IDs.

* --ids (required): Comma-separated list of cluster IDs to keep (e.g., 2523,210).

* --out (.h5ad): Path to save the output subset AnnData file.

2. Outputs

* A subset .h5ad file that contains only the cells belonging to the specified clusters.

* A new column subtype is added to obs, where each selected cluster is labeled (e.g., cluster_2523, cluster_210).

## Program 5: s5_subh5adCheck.py

This script provides a quick diagnostic summary of a subset .h5ad file. 

It automatically detects the cluster annotation column in obs, counts how many cells belong to specific clusters (2523 and 210), and optionally checks these counts restricted to the Cerebral cortex region. 

Finally, it reports the file size.

1. Parameters

This script does not take command-line arguments; instead, key inputs are hard-coded:

* PATH (string): Path to the .h5ad file to inspect.

* obs columns (inside the file): The script automatically scans for one of these cluster-like columns: ["cluster", "subtype", "cluster_alias", "cluster_label", "label"]. The first match is used.

* anatomical_division_label (optional column in obs): If present, the script further restricts analysis to cells from the Cerebral cortex only.

2. Outputs

Printed summary to console, including:

* Total number of cells in the dataset.

* Number of cells in cluster_2523 and cluster_210.

* Name of the cluster column used and example values.

* If available: counts restricted to the Cerebral cortex.

* File size of the .h5ad in human-readable units (e.g., MB, GB).

![[fig11]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig11.png?raw=true)

The results cells from these two subcluster are not all located in cerebral cortex, so I will next the next Program 6 to grab cells only in cerebral cortex.

## Program 6: s6_subh5adRegion.py

This script filters a given .h5ad dataset to retain only cells from a specific anatomical region. 

The region is defined by matching values in a chosen metadata (obs) column, typically anatomical_division_label. 

1. Parameters

* --in (string, required): Path to the input .h5ad file.

* --out (string, required): Path to save the filtered .h5ad file.

* --region (string, required): Exact region name to filter (e.g., "Cerebral cortex").

* --region-col (string, optional, default=anatomical_division_label): Column in obs that stores region annotations. If missing, the script will raise an error.

2. Outputs

A new .h5ad file (--out) containing only cells that belong to the specified region.

## Program 7: s7_makeUnique.py

This script ensures that gene names (var_names) and cell barcodes (obs_names) in an AnnData object are unique. 

While this functionality is already included in the downstream Program 8, this script can be used independently if you only need to enforce uniqueness of IDs without running the full pipeline.

1. Parameters

* --infile (required): Path to the input .h5ad file.

* --outfile (required): Path to the output .h5ad file, where the unique IDs will be saved.

2. Output

* A new .h5ad file with unique gene names and cell IDs.

* Console messages reporting how many duplicate gene names and cell IDs were detected.

## Program 8: s8_cellGeneCrop.py

This script builds a small, reproducible “hello-world” AnnData slice from a large .h5ad. It is designed to give you a compact subset (e.g., a few hundred–thousand cells and selected genes) that preserves basic signal structure for quick prototyping and pre-training. Pipeline stages implemented in the code:

* Load the input .h5ad. If layers["counts"] exists it is used as the numeric basis; otherwise adata.X is used.

* De-duplicate IDs: make cell IDs (obs_names) and gene names (var_names) unique, or collapse/drop duplicates according to user options.

* Optionally drop all-zero genes/cells.

* Summarize sparsity (before): shape, non-zero fraction, median non-zeros per cell/gene.

* Stratified cell sampling:

	* Compute per-cell non-zero counts.

	* Split cells into strata quantile bins (to cover low/medium/high depth cells).

	* Sample roughly equally from each bin until reaching --cells (seeded for reproducibility).

	* Gene selection from the sampled cells:

	* Compute per-gene detection fraction (share of sampled cells where gene > 0) and standard deviation.

	* Keep genes with detection ≥ min_gene_detect_frac. If not enough, back-off to the top genes by detection fraction.

	* Draw weighted, no-replacement gene samples up to --genes, with weight. This biases toward informative, variable genes.

$$
w_g \propto (\text{detect}_g)^{\alpha_{\text{detect}}} \cdot (\text{std}_g)^{\beta_{\text{std}}}
$$

1. Parameters

* -i, --input (required): Input .h5ad. If layers["counts"] exists it’s used; otherwise X is used.

* -o, --output (required): Output .h5ad (gzipped).

* --report_csv (optional): Path to write a small CSV with before/after sparsity metrics.

* --cells (int, default=200): Target number of cells to sample. If larger than available, all cells are used. (Note: setting --cells to a very large value, e.g., 1,000,000, simply keeps all cells; it does not break the code.)

* --genes (int, default=500): Target number of genes to select. If fewer candidates remain, you’ll get as many as possible.

* --seed (int, default=1): RNG seed for reproducibility.

* --strata (int, default=4): Number of quantile bins for stratified cell sampling. 1 ≈ simple random sample; larger values better preserve depth diversity.

* --min_gene_detect_frac (float, default=0.05): Minimum detection fraction to admit a gene to the candidate pool.

* --alpha_detect (float, default=1.0): Exponent on detection fraction in the gene-weight formula. Higher → favor widely detected genes.

* --beta_std (float, default=0.5): Exponent on per-gene std in the gene-weight formula. Higher → favor variable genes.

* --gene_dedupe_mode {"make_unique","collapse"} (default: make_unique):

	* make_unique: Scanpy will suffix duplicates (Gene, Gene-1, …).

	* collapse: Sum duplicate-named gene columns into one, update var accordingly (applied consistently to layers['counts']/X).

* --cell_dedupe_mode {"make_unique","drop"} (default: make_unique):

	* make_unique: suffix duplicate cell IDs.

	* drop: keep the first occurrence, drop later duplicates.

	* --drop_allzero_genes (flag): Drop any gene whose column is all zeros.

	* --drop_allzero_cells (flag): Drop any cell whose row is all zeros.


2. Output

* Gzipped .h5ad at --output, containing only the selected cells × genes.

* If input had layers['counts'], it remains aligned and subset.

* adata.uns includes:

	* crop_params: all parameters used, input_basis ("layers['counts']" or "X"), seed, dedupe modes, etc.

	* sparsity_summary: dictionaries for before and after with
shape, nonzero_fraction, median_nonzero_per_cell, median_nonzero_per_gene.

	* crop_indices: the selected cell and gene indices relative to the input (as lists).

* Optional CSV (--report_csv): two rows (before, after) with the same sparsity metrics.

* Stdout logs: sizes chosen, basic checks, and any drops applied.

3. Example usage

```
python crop_hello_world.py \
  -i big_atlas.h5ad \
  -o subset_hello.h5ad \
  --report_csv subset_metrics.csv \
  --cells 1000000 --genes 2000 \
  --strata 4 --min_gene_detect_frac 0.05 \
  --alpha_detect 1.0 --beta_std 0.5 \
  --gene_dedupe_mode make_unique \
  --cell_dedupe_mode make_unique \
  --drop_allzero_genes --drop_allzero_cells \
  --seed 42
```

## Program 9: s9_fModelMain.py

This script performs pre-training of a Transformer-based foundation model on a single-cell RNA-seq dataset.

The continuous gene expression values are normalized, log-transformed, discretized into bins (quantization), and then masked at random.

The model learns to reconstruct the masked tokens using an encoder-only Transformer, similar to masked language modeling in NLP.

The script logs training/validation performance, evaluates perplexity and masked prediction accuracy, and saves checkpoints, metrics, and diagnostic plots.

1. Parameters

The script uses hard-coded configuration values at the top of the file:

* DATA_PATH: Path to input .h5ad file (AnnData object with cell × gene matrix).

* CHECKPOINT_DIR / LOG_DIR / FIG_DIR: Output directories for model weights, logs, and figures.

* SEED: Random seed for reproducibility.

* USE_LOG1P: Whether to apply log1p transformation after normalization.

* NUM_BINS: Number of discrete bins for expression quantization.

* BATCH_SIZE / VAL_BATCH: Batch sizes for training and validation.

* EPOCHS: Number of training epochs.

* EMB_DIM / NHEAD / NLAYERS / DROPOUT: Transformer architecture hyperparameters.

* LR / WEIGHT_DECAY: Optimizer learning rate and weight decay.

* VAL_RATIO: Fraction of cells held out for validation.

* MASK_RATIO: Fraction of genes masked per cell.

* NONZERO_FRAC: Proportion of masked positions forced to come from nonzero expression values.

2. Output

* Checkpoints (CHECKPOINT_DIR/hello_model_best.pt, hello_model_last.pt): Best-performing model on validation loss, and final epoch model.

* Binning metadata (binning_meta.npz): Stores per-gene bin edges, gene order, and preprocessing settings for downstream reproducibility.

* Logs (LOG_DIR/train.log): Training and validation progress with losses, perplexity, and masked accuracy.

* Training metrics CSV (LOG_DIR/training_metrics.csv): Epoch-wise values of train/validation loss, perplexity, and accuracy.

* Figures (FIG_DIR/): Line plots of train loss, val loss, val perplexity, and val masked accuracy.

* Quick demo (console log): Example masked predictions for one validation cell, showing top-k guesses vs. ground truth.

3. Usage in the Project

```
SEED       = 42
USE_LOG1P  = True
NUM_BINS   = 20
BATCH_SIZE = 64
VAL_BATCH  = 128
EPOCHS     = 30
EMB_DIM    = 64
NHEAD      = 4
NLAYERS    = 3
DROPOUT    = 0.1
LR         = 5e-4
WEIGHT_DECAY = 1e-2
VAL_RATIO  = 0.2
MASK_RATIO = 0.15
NONZERO_FRAC = 0.5
```

## Program 10: s10_probeEval.py

This script evaluates my pre-trained Transformer by training a linear probe (logistic regression) on top of the model’s frozen embeddings. Here are the steps:

* Loads an AnnData .h5ad dataset and applies the same preprocessing as in pre-training (normalize_total, optional log1p).

* Loads the saved tokenizer/binning metadata (binning_meta.npz) to reorder genes and re-bin the expressions exactly as during pre-train.

* Loads the pre-trained Transformer checkpoint, runs encode() to extract per-cell embeddings (mean-pooled across genes).

* Splits cells into train/test and fits a logistic regression classifier to predict Excitatory (1) vs Inhibitory (0) labels.

* Reports Accuracy, F1, ROC-AUC, saves a confusion matrix, and produces PCA and ROC plots.

* Saves embeddings, labels, the trained probe, and a run metadata JSON for reproducibility.

1. Parameters

* --adata (required): Path to the .h5ad. Must contain the same genes as used in pre-train.

* --checkpoint (required): Path to pre-trained Transformer weights (e.g., hello_model_best.pt).

* --binmeta (required): Path to binning_meta.npz saved in pre-train (contains edges, gene_names, num_bins, etc.). Used to re-bin and align gene order.

* --use_log1p: Apply log1p after normalization. Must match pre-train.

* --num_bins (default 20): Number of bins; must equal the value used in pre-train.

* --batch_size (default 128): Batch size for embedding extraction.

* --emb_dim, --nhead, --nlayers, --dropout, --mask_token_id: Model architecture hyperparameters; must match the pre-trained checkpoint.

* --val_ratio (default 0.2): Fraction of cells held out for probe testing (stratified).

* --outdir (default probe_outputs): Output directory for all artifacts.

* --label_col (default subtype): Column in adata.obs containing cell labels to be mapped.

* --exc_labels (default cluster_2523): Comma-separated values in label_col mapped to 1 (Excitatory).

* --inh_labels (default cluster_210): Comma-separated values in label_col mapped to 0 (Inhibitory).

2. Outpus

All files are written to --outdir:

* embeddings.npy: 2D NumPy array [n_cells × emb_dim] with mean-pooled cell embeddings from the frozen Transformer.

* labels.npy: 1D NumPy array [n_cells] of binary labels (0=Inhibitory, 1=Excitatory).

* cells.tsv: TSV mapping of cell (obs name) to numeric label.

* logreg_probe.pkl: Trained logistic regression probe (saved with joblib).

* confusion_matrix.csv: 2×2 table with counts for true/predicted classes.

* pca_probe.png: PCA (2D) visualization of embeddings colored by label.

* roc_probe.png: ROC curve with AUC for the held-out test set.

* run_meta.json: JSON capturing all key settings and metrics (paths, binning, model hyperparameters, accuracy/F1/ROC-AUC, n_cells/n_genes).

3. Usage in the Project

```
python probeEval.py \
  --adata subset_2523_210_Cerebral-cortex.screened.h5ad \
  --checkpoint ckpts_2523_210_1/hello_model_best.pt \
  --binmeta ckpts_2523_210_1/binning_meta.npz \
  --use_log1p \
  --num_bins 20 \
  --batch_size 128 \
  --emb_dim 64 \
  --nhead 4 \
  --nlayers 3 \
  --dropout 0.1 \
  --val_ratio 0.2 \
  --label_col subtype \
  --exc_labels cluster_2523 \
  --inh_labels cluster_210 \
  --outdir probe_outputs_v1
```

## Program 11: s11_afterProbeVis.py

This script performs additional visualization of the learned cell embeddings after the probe evaluation. It applies two common dimensionality reduction methods—t-SNE and UMAP—to project the high-dimensional embeddings into 2D space. 

Cells are colored according to their neuronal type labels (Excitatory vs. Inhibitory), allowing users to visually inspect whether the embeddings separate biological subtypes.

1. Parameters

* embeddings.npy: A NumPy array of shape (cells × dimensions), containing the embeddings exported by the probe script.

* labels.npy: A NumPy array of shape (cells,) with binary class labels (0 = Inhibitory neurons, 1 = Excitatory neurons).

* perplexity (default=30): t-SNE parameter that balances local vs. global structure.

* random_state (default=42): Random seed to ensure reproducibility.

* n_components (default=2): Number of output dimensions (2D projection).

2. Outputs

* tsne_probe.png: 2D scatter plot of embeddings reduced by t-SNE, colored by neuronal labels.

* umap_probe.png: 2D scatter plot of embeddings reduced by UMAP, colored by neuronal labels.

Clear separation of excitatory (red) and inhibitory (blue) neurons suggests that the pretrained embeddings capture biologically relevant signals.

## Pre-train Level Results

![[fig4]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig4.png?raw=true)

![[fig5]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig5.png?raw=true)

![[fig6]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig6.png?raw=true)

1. Results

The validation perplexity started around 8.5 and decreased steadily across training epochs, reaching about 6.3 by epoch 30.

This shows that the model is learning to reconstruct masked tokens more effectively over time, capturing statistical patterns in gene expression bins (tokens).

The curve shows that improvement slows down after epoch 20, suggesting the model approaches a plateau.

Yet, perplexity alone does not confirm biological semantics—it only measures reconstruction fit on the token distribution.

Compared to language models, a perplexity of 6 is moderate: the model can predict masked bins better than chance, but not yet highly confidently.

Validation and training losses track closely, suggesting no severe overfitting in this setup.

2. Discussion

To further reduce perplexity and enhance representation quality:

* Include more clusters or additional cells to expose the model to a wider range of biological variation.

* Use GPU acceleration for faster training would allow more epochs and larger models.

* larger embedding dimension or more transformer layers may help capture more complex dependencies.

## Probe Level Results

1. Results

![fig7](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig7.png?raw=true)

True Inhibitory (0): 94 correctly classified, 13 misclassified as Excitatory. True Excitatory (1): 91 correctly classified, 33 misclassified as Inhibitory.

The classifier achieves higher precision for Excitatory (1) but slightly better recall for Inhibitory (0).

Overall accuracy ≈ 0.80: This shows that the probe captures meaningful biological signals from the embeddings.

Excitatory cells are more likely to be confused with Inhibitory cells (33 misclassified) than vice versa (13 misclassified). This suggests that Excitatory embeddings overlap more with Inhibitory in the feature space.

![fig8](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig8.png?raw=true)

The PCA plot shows that inhibitory and excitatory neurons overlap almost completely in the first two principal components.

This indicates that the main axes of variance in the embeddings are not strongly aligned with the Exc/Inh distinction. 

PCA mainly captures global variance, so the separation is limited.

![[fig9]](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig9.png?raw=true)

UMAP reveals more fine-grained local organization, showing partially mixed but slightly structured separation.

There are small regions enriched for excitatory or inhibitory cells, but still with strong overlap.

This suggests that the embeddings contain subtle but not dominant biological signals.

![fig10](https://github.com/DZBohan/neuron_type_foundation_model/blob/main/images/fig10.png?raw=true)

t-SNE highlights localized clusters where inhibitory and excitatory neurons show clearer separation than in PCA.

Compared to UMAP, the Exc/Inh distinction is slightly stronger, but overlap remains.

Across methods, linear PCA fails to reveal separation, while nonlinear methods (UMAP, t-SNE) extract weak subtype structure.

This matches the quantitative results (Accuracy ≈ 0.80, ROC-AUC ≈ 0.86): the signal exists but is not fully disentangled in the embedding space.

The model has learned some biologically meaningful representations, but the model is in its initial stage.

2. Discussion

Ways to Improve model performance:

* Pre-train on a larger dataset including more neuronal diversity.

* Increase model capacity (larger embedding dimension, more layers, GPU training).