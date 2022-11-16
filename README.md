# Gene-Interaction Graph Neural Network to Predict Cancer Drug Response

## Introduction
This repository contains the process and the final code for my master thesis.

## Installation

```bash
conda create ...
conda activate ...
pip install ...
```

## Contents
### Notebooks

| Notebook | Content |
| -------- | ------- |
| `02_GDSC_map_GenExpr.ipynb` | Contains the code for the creation of the base dataset containing gene expressions for cell-line drug combinations. |
| `03_GDSC_map_CNV.ipynb` | Contains the code for the creation of the base dataset containing gistic and picnic copy numbers for cell-line drug combinations. |
| `04_GDSC_map_mutations.ipynb` | |
| `05_DrugFeatures.ipynb` | |
| `06_create_base_dataset.ipynb` | |
| `07_Linking.ipynb` | Contains the code for the creation of the graph using the STRING database. |
| `07_v1_2_get_linking_dataset.ipynb` | Creates the graph per cell-line with all 4 node features (gene expr, cnv gistic, cnv picnic and mutation). |
| `07_v2_graph_dataset.ipynb` | Used only the gene-gene tuples with a `combined_score` value of more then 950. Ended up with only 458 genes per cell-line for now (instead of 858 as of before). | 

### Todos 

- [x] fix error with mutations dataset 
- [x] include mutation features in tensor
- [x] start building building bi-modal network structure and build simple NN
- [x] correct `DataLoader` to access cell-line-gene-drug-ic50 tuple correctly
- [x] shuffle
- [ ] Run `TabGraph` with choosing an appropriate GNN layer type
- [ ] For `GraphTab` and `GraphGraph` use the `combined_score` column to sparse down the connections between the genes by using an appropriate threshold, e.g. `0.95*1_000`
- [ ] include GDSC 1 data as well; check shift in ln(IC50)'s and think about strategy to meaningful combine both in single dataset
  - [ ] Combine both GDSC1 and GDSC2 in an complete dataset to increase training data
- [ ] once all models are running, convert to `.py` files instead of notebooks
- [ ] run GNNExplainer on the graph branches of the bi-modal networks

__Networks__:

- [x] Build baseline models using
  - [x] Regression (see `16_simple_regression.ipynb`)
  - [x] Random forest (see `16_simple_regression.ipynb`)
  - [x] MPL (see `16_simple_regression.ipynb`)
- [x] `Tab-Tab`: General FFNN using tabular data for drugs and cell-lines _(refs: [MOLI-2019](https://academic.oup.com/bioinformatics/article/35/14/i501/5529255) ([code](https://github.com/hosseinshn/MOLI)))_
- [ ] `Graph-Tab`: Cell-line branch by GNN, drug branch by tabular data _(refs: [GraphGCN-2021]([file:///Users/cwoest/Downloads/mathematics-09-00772%20(7).pdf](https://www.mdpi.com/2227-7390/9/7/772/htm)) ([code](https://github.com/BML-cbnu/DrugGCN)))_
- [ ] `Tab-Graph`: Cell-line branch by tabular data, drug branch by GNN _(refs: [DrugDRP-2022](https://pubmed.ncbi.nlm.nih.gov/33606633/) ([code](https://github.com/hauldhut/GraphDRP)))_
- [ ] `Graph-Graph`: Replace both cell-line branch & drug branch by GNNs _(refs: [TGSA-2022](https://academic.oup.com/bioinformatics/article/38/2/461/6374919) ([code](https://github.com/violet-sto/TGSA)))_
- [ ] Implement [`GNNExplainer-2019`](https://arxiv.org/abs/1903.03894) ([code](https://github.com/RexYing/gnn-model-explainer)) on the above network methods
  - [x] Run on `Tab-Tab` (see `12_v1_TabTab.ipynb`)
  - [ ] Run on `Graph-Tab`
  - [ ] Run on `Tab-Graph`
  - [ ] Run on `Graph-Graph`
  - [ ] Convert to non-notebook py code
    - [ ] include setting of args from terminal