# Graph Neural Networks for Drug Response Prediction in Cancer

## :bulb: Introduction
This repository contains the process and the final code for my master thesis "_Gene-Interaction Graph Neural Network to Predict Cancer Drug Response_".

## :computer: Environment Setup
### Using conda
To create the virtual environment via `conda` run
```bash
# Option 1: by using the environment.yml file
conda env create -n ENVNAME --file environment.yml

# Option 2: by using the requirement.txt file
conda create -n ENVNAME --file requirements.txt
```
Now activate the environment.
```bash
conda activate ENVNAME
```

### Using pip

- [ ] This may not work yet. I have only tested the conda method yet.

To create and activate the virtual environment via `pip` run
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## :arrow_double_down: Download Raw Datasets
To download the raw datasets which can be used to create the training datasets run
```python
# Choose the following two paths by yourself! Here are just examples.
RAW_PATH = '../../datatest/raw/'
PROCESSED_PATH = '../../datatest/processed/'

from pathlib import Path
Path(RAW_PATH).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_PATH).mkdir(parents=True, exist_ok=True)

from src.preprocess.build_features import Processor
processor = Processor(raw_path=RAW_PATH,
                      processed_path=PROCESSED_PATH)

processor.download_raw_datasets()
```

For `raw_path` and `processed_path` we recommend to choose a folder outside of this repository since some files are very large (>100MB). 
- For details on the dataset sizes and contents read [data/README.md](data/README.md). 
- For an example of how to run this code refer to [notebooks/download_raw_datasets.ipynb](notebooks/download_raw_datasets.ipynb).

## :books: Contents
### Notebooks

| Notebook | Content |
| -------- | ------- |
| [`02_GDSC_map_GenExpr.ipynb`](02_GDSC_map_GenExpr.ipynb) | Contains the code for the creation of the base dataset containing gene expressions for cell-line drug combinations. |
| [`03_GDSC_map_CNV.ipynb`](03_GDSC_map_CNV.ipynb) | Contains the code for the creation of the base dataset containing gistic and picnic copy numbers for cell-line drug combinations. |
| [`04_GDSC_map_mutations.ipynb`](04_GDSC_map_mutations.ipynb) | |
| [`05_DrugFeatures.ipynb`](05_DrugFeatures.ipynb) | |
| [`06_create_base_dataset.ipynb`](06_create_base_dataset.ipynb) | |
| [`07_Linking.ipynb`](07_Linking.ipynb) | Contains the code for the creation of the graph using the STRING database. |
| [`07_v1_2_get_linking_dataset.ipynb`](07_v1_2_get_linking_dataset.ipynb) | Creates the graph per cell-line with all 4 node features (gene expr, cnv gistic, cnv picnic and mutation). Topology per cell-line graph is `Data(x=[858, 4], edge_index=[2, 83126])`. |
| [`07_v2_graph_dataset.ipynb`](07_v2_graph_dataset.ipynb) | Used only the gene-gene tuples with a `combined_score` value of more then 950. Ended up with only 458 genes per cell-line for now (instead of 858 as of before). Filtered 1st by the `combined_score` and than by the landmark genes. Topology per cell-line graph is `Data(x=[458, 4], edge_index=[2, 4760])`. |
| [`07_v3_graph_dataset.ipynb`](07_v3_graph_dataset.ipynb) | First select only the landmark genes from the protein-protein interaction table. Then tune the threshold for the `combined_score` column according to how many unique genes would be left. |
| [`11_v1_GraphTab_sparse_1.ipynb`](11_v1_GraphTab_sparse_1.ipynb) | Used the dataset from [`07_v2_graph_dataset.ipynb`](07_v2_graph_dataset.ipynb) having topology per cell-line graph of `Data(x=[458, 4], edge_index=[2, 4760])` |
| [`11_v1_GraphTab_nonsparse.ipynb`](11_v1_GraphTab_nonsparse.ipynb) | Used the dataset from [`07_v1_2_get_linking_dataset.ipynb`](07_v1_2_get_linking_dataset.ipynb) having topology per cell-line graph of `Data(x=[858, 4], edge_index=[2, 83126])`. Took too long per epoch, which is why different approaches needed to be found to sparse the number of edges in the graph (see `combined_score` approach). | 

### Todos 

- [x] fix error with mutations dataset 
- [x] include mutation features in tensor
- [x] start building building bi-modal network structure and build simple NN
- [x] correct `DataLoader` to access cell-line-gene-drug-ic50 tuple correctly
- [x] shuffle
- [x] Run `TabGraph` with choosing an appropriate GNN layer type (see [`12_v1_TabTab.ipynb`](12_v1_TabTab.ipynb))
- [x] For `GraphTab` and `GraphGraph` use the `combined_score` column to sparse down the connections between the genes by using an appropriate threshold, e.g. `0.95*1_000` (see [`07_v2_graph_dataset.ipynb`](07_v2_graph_dataset.ipynb))
  - problem: number of genes got decreased to 484 from 858
  - [ ] Filter first by landmark genes and than by the `combined_score` (and not the other way around as done in [`07_v2_graph_dataset.ipynb`](07_v2_graph_dataset.ipynb)) (see TODO)
- [ ] Save also the other performance metrics per epoch (r, r2, mae, rmse)
- [ ] include GDSC 1 data as well; check shift in ln(IC50)'s and think about strategy to meaningful combine both in single dataset (see TODO)
  - [ ] Combine both GDSC1 and GDSC2 in an complete dataset to increase training data (see TODO)
- [ ] Parallelize code using `num_workers > 0`
  - [ ] once all models are running, convert to `.py` files instead of notebooks
- [ ] run GNNExplainer on the graph branches of the bi-modal networks

__Networks__:

- [x] Build baseline models using
  - [x] Regression (see [`16_simple_regression.ipynb`](16_simple_regression.ipynb))
  - [x] Random forest (see [`16_simple_regression.ipynb`](16_simple_regression.ipynb))
  - [x] MPL (see [`16_simple_regression.ipynb`](16_simple_regression.ipynb))
- [x] `Tab-Tab`: General FFNN using tabular data for drugs and cell-lines _(refs: [MOLI-2019](https://academic.oup.com/bioinformatics/article/35/14/i501/5529255) ([code](https://github.com/hosseinshn/MOLI)))_
- [ ] `Graph-Tab`: Cell-line branch by GNN, drug branch by tabular data _(refs: [GraphGCN-2021]([file:///Users/cwoest/Downloads/mathematics-09-00772%20(7).pdf](https://www.mdpi.com/2227-7390/9/7/772/htm)) ([code](https://github.com/BML-cbnu/DrugGCN)))_
- [ ] `Tab-Graph`: Cell-line branch by tabular data, drug branch by GNN _(refs: [DrugDRP-2022](https://pubmed.ncbi.nlm.nih.gov/33606633/) ([code](https://github.com/hauldhut/GraphDRP)))_
- [ ] `Graph-Graph`: Replace both cell-line branch & drug branch by GNNs _(refs: [TGSA-2022](https://academic.oup.com/bioinformatics/article/38/2/461/6374919) ([code](https://github.com/violet-sto/TGSA)))_
- [ ] Implement [`GNNExplainer-2019`](https://arxiv.org/abs/1903.03894) ([code](https://github.com/RexYing/gnn-model-explainer)) on the above network methods
  - [x] Run on `Tab-Tab` (see [`12_v1_TabTab.ipynb`](12_v1_TabTab.ipynb))
  - [ ] Run on `Graph-Tab` (see TODO)
  - [x] Run on `Tab-Graph` (see [`13_v1_TabGraph.ipynb`](13_v1_TabGraph.ipynb))
  - [ ] Run on `Graph-Graph`
  - [ ] Convert to non-notebook `.py` code
    - [ ] include setting of args from terminal

## Questions 
- [ ] Why is each cell-line graph still `G.is_undirected()==False` even though I included manual checks for it? Does this matter?