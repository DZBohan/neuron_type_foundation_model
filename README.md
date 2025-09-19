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

## Program 9: s9_fModelMain.py

This script performs pre-training of a Transformer-based foundation model on a single-cell RNA-seq dataset.

The continuous gene expression values are normalized, log-transformed, discretized into bins (quantization), and then masked at random.

The model learns to reconstruct the masked tokens using an encoder-only Transformer, similar to masked language modeling in NLP.

The script logs training/validation performance, evaluates perplexity and masked prediction accuracy, and saves checkpoints, metrics, and diagnostic plots.

1. Parameters

The script uses hard-coded configuration values at the top of the file (the “Config Zone”):

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

## Program 10: s10_probeEval.py

This script evaluates my pre-trained Transformer by training a linear probe (logistic regression) on top of the model’s frozen embeddings. Here are the steps:

* Loads an AnnData .h5ad dataset and applies the same preprocessing as in pre-training (normalize_total, optional log1p).

* Loads the saved tokenizer/binning metadata (binning_meta.npz) to reorder genes and re-bin the expressions exactly as during pre-train.

* Loads the pre-trained Transformer checkpoint, runs encode() to extract per-cell embeddings (mean-pooled across genes).

* Splits cells into train/test and fits a logistic regression classifier to predict Excitatory (1) vs Inhibitory (0) labels.

* Reports Accuracy, F1, ROC-AUC, saves a confusion matrix, and produces PCA and ROC plots.

* Saves embeddings, labels, the trained probe, and a run metadata JSON for reproducibility.

1. Parameters

--adata (required): Path to the .h5ad. Must contain the same genes as used in pre-train.

--checkpoint (required): Path to pre-trained Transformer weights (e.g., hello_model_best.pt).

--binmeta (required): Path to binning_meta.npz saved in pre-train (contains edges, gene_names, num_bins, etc.). Used to re-bin and align gene order.

--use_log1p: Apply log1p after normalization. Must match pre-train.

--num_bins (default 20): Number of bins; must equal the value used in pre-train.

--batch_size (default 128): Batch size for embedding extraction.

--emb_dim, --nhead, --nlayers, --dropout, --mask_token_id: Model architecture hyperparameters; must match the pre-trained checkpoint.

--val_ratio (default 0.2): Fraction of cells held out for probe testing (stratified).

--outdir (default probe_outputs): Output directory for all artifacts.

--label_col (default subtype): Column in adata.obs containing cell labels to be mapped.

--exc_labels (default cluster_2523): Comma-separated values in label_col mapped to 1 (Excitatory).

--inh_labels (default cluster_210): Comma-separated values in label_col mapped to 0 (Inhibitory).

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

fig7

True Inhibitory (0): 94 correctly classified, 13 misclassified as Excitatory. True Excitatory (1): 91 correctly classified, 33 misclassified as Inhibitory.

The classifier achieves higher precision for Excitatory (1) but slightly better recall for Inhibitory (0).

Overall accuracy ≈ 0.80: This shows that the probe captures meaningful biological signals from the embeddings.

Excitatory cells are more likely to be confused with Inhibitory cells (33 misclassified) than vice versa (13 misclassified). This suggests that Excitatory embeddings overlap more with Inhibitory in the feature space.

fig8

The PCA plot shows that inhibitory and excitatory neurons overlap almost completely in the first two principal components.

This indicates that the main axes of variance in the embeddings are not strongly aligned with the Exc/Inh distinction. 

PCA mainly captures global variance, so the separation is limited.

fig9

UMAP reveals more fine-grained local organization, showing partially mixed but slightly structured separation.

There are small regions enriched for excitatory or inhibitory cells, but still with strong overlap.

This suggests that the embeddings contain subtle but not dominant biological signals.

fig10

t-SNE highlights localized clusters where inhibitory and excitatory neurons show clearer separation than in PCA.

Compared to UMAP, the Exc/Inh distinction is slightly stronger, but overlap remains.

Across methods, linear PCA fails to reveal separation, while nonlinear methods (UMAP, t-SNE) extract weak subtype structure.

This matches the quantitative results (Accuracy ≈ 0.80, ROC-AUC ≈ 0.86): the signal exists but is not fully disentangled in the embedding space.

The model has learned some biologically meaningful representations, but the model is in its initial stage.

2. Discussion

Ways to Improve model performance:

* Pre-train on a larger dataset including more neuronal diversity.

* Increase model capacity (larger embedding dimension, more layers, GPU training).