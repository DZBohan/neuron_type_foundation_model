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

[fig1]

Progect: Transcriptomic diversity of cell types in adult human brain

[fig2]

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

[fig4]

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

Program 2: 
