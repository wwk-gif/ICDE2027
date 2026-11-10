# ICDE2027 — MAGNet

Code and input datasets for **MAGNet: Coordinated Representation Learning for Heterogeneous Spatial Transcriptomics Integration**.

## Contents

- `MAGNet/`: model, training pipelines, section agents, semantic sampler, graph construction, alignment, clustering and evaluation.
- `Config/`: osmFISH, olfactory-bulb (`MOB.yaml`) and MERFISH configurations.
- `datasets/`: the eight H5AD input files used for these three tasks.
- `Reproduction_Notebook/osmFISH.ipynb`: portable osmFISH analysis notebook.
- `requirement.txt`: Python dependency versions from the supplied development environment.

## Installation

Use an isolated Python environment, install PyTorch and its compatible PyTorch Geometric extensions for your CUDA version, then install the listed Python dependencies:

```bash
pip install -r requirement.txt
```

The clustering and semantic-sampling code uses R through `rpy2`. Install R and its `mclust` package first:

```r
install.packages("mclust")
```

Configure `R_HOME` for your own installation if R is not on the system path. The notebooks use project-relative dataset paths.

## Training entry point

Use `MAGNet.SC_BC_pipeline` for the ICDE workload. The existing API spells its training method `trian()`:

```python
import MAGNet as TOOLS

net = TOOLS.SC_BC_pipeline(
    adata,
    edge_index=edge_index,
    num_clusters=num_clusters,
    device=device,
    config=config,
    imputation=False,
)
net.trian()
embedding, reconstruction = net.process()
```

Prepare `adata.obsm["X_pca"]` with the configured input features and `adata.obs["slice_id"]` with consecutive integer section IDs. Build `edge_index` with the supplied graph-construction functions. On osmFISH the feature slot contains the scaled targeted-panel features and one section uses ID 0.

`SC_BC_pipeline` reads both `train.topk_neighs` and `train.topk_neighs_inter`. The supplied YAML files do not contain the latter key. The single-section osmFISH notebook explicitly uses zero cross-section neighbors. For joint fitting, set `topk_neighs_inter` and any run-specific preprocessing/graph overrides to the executed experiment configuration before starting training.

Launch Jupyter from this directory and open `Reproduction_Notebook/osmFISH.ipynb`. This notebook retains the source preprocessing and analyses, with portable imports/paths, cleared outputs, and the author-selected `SC_BC_pipeline` entry point.

## Datasets

| Task | Files | Source study |
|---|---|---|
| osmFISH somatosensory cortex | `datasets/osmFISH/adata.h5ad` | Codeluppi et al., Nature Methods (2018) |
| Olfactory bulb, Stereo-seq and Slide-seqV2 | `datasets/slideseq/MOB_Stereo.h5ad`, `MOB_Slide.h5ad` | Chen et al., Cell (2022); Stickels et al., Nature Biotechnology (2021) |
| MERFISH preoptic-area series | five `datasets/merfish/hypothalamus-*.h5ad` files | Moffitt et al., Science (2018) |

`datasets/manifest.json` records file sizes, SHA-256 checksums and the names of the source archives. The largest file is approximately 57.7 MB. Download the repository as a ZIP or clone it to obtain the datasets together with the code.

The source module names include historical identifiers such as `MACWT_model`; the public package import is `MAGNet`. The core Python modules and the supplied YAML configurations are copied byte-for-byte from the author's local MAGNet project. Packaging checks include Python syntax, notebook-cell syntax and dataset checksum validation. GPU training has not been rerun as part of this upload preparation.
