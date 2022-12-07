
# :calendar: Todos 

- [ ] Run baseline models as comparison for the chosen combined_score_thresh
    - [ ] Predicting the mean
    - [ ] Regression
    - [ ] MLP
    - [ ] Random Forest 
    - [ ] SVM
- [ ] Compare predicted with observed ic50 values for each model on the test set
- [ ] For the cell-GNN try and monitor multiple architecture
    - [x] `GCNConv` vs `GATConv`
    - [ ] `global_mean_pool` vs `global_max_pool`
    - [ ] Number of GNN layers (hops)
    - [ ] Choice of `combined_score_thresh` and therefore the number of genes in the graph
- [ ] Calculate the average predicted ic50 values for each drug. Choose the top 10 and bottom 10 drugs.
- [ ] Run crossvalidation and choose for example 5 different random seeds. Monitor average performance over set of seeds.
- [ ] Check other performance metric [spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)