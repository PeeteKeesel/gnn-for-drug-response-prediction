import pandas as pd
import torch

from torch_geometric.data import Data


# Holding the cell line as key and the created graph as value.
cell_line_graph = {}

"""
For each cell-line the objective is to create a graph. This graph has the same nodes and edges
per cell-line, only the feature values differ.

Given:
    (UNIQ_LANDMARK_GENES): DataFrame containing the landmark genes.
    (gene_symbol_tuples): All gene neihbor tuples from the STRING database. Sparsed down
                          by only taking these which are in the landmark gene set.
    (uniq_cell_line_names): List of unique cell-line names which are in all node feature datasets. 
    (set_of_gene_symbols): Set of gene symbols which are in all node feature datasets and
                           which are thus building the same graph for each cell-line.
                           Only different in node feature values.
    (gene_expr): DataFrame containing the gene expression for all cell line names.
    (cnv_gis): DataFrame containing the gistic copy numbers for all cell line names.
    (cnv_pic): DataFrame containing the picnic copy numbers for all cell line names.
"""
gene_symbol_tuples = NotImplementedError
uniq_cell_line_names = NotImplementedError
set_of_gene_symbols = NotImplementedError
gene_expr = NotImplementedError
cnv_gis = NotImplementedError
cnv_pic = NotImplementedError

# Iterate over the cell-line names.
for cln in uniq_cell_line_names:

    # Obtain corresponding row for each node feature.
    row_gexpr = gene_expr.loc[gene_expr.CELL_LINE_NAME==cln]
    row_cnvg = cnv_gis.loc[cnv_gis.CELL_LINE_NAME==cln]
    row_cnvp = cnv_pic.loc[cnv_pic.CELL_LINE_NAME==cln]

    # Assert that its only one row. 
    assert row_gexpr.shape[0] == row_cnvg.shape[0] == row_cnvp.shape[0] == 1, \
        "InputError: There are multiple rows for cell line {cln}!"

    # TODO: Check that the gene column are the same.
    # Select only the gene columns.
    vals_gexpr = row_gexpr[set_of_gene_symbols]
    vals_cnvg = row_cnvg[set_of_gene_symbols]
    vals_cnvp = row_cnvp[set_of_gene_symbols]

    # Order the gene columns such that they are in the same order for all datasets.
    vals_cnvg = vals_cnvg[vals_gexpr.columns]
    vals_cnvp = vals_cnvp[vals_gexpr.columns]

    # Get the node feature vector for each gene.
    node_features = list(map(list, zip(*vals_gexpr.values, *vals_cnvg.values, *vals_cnvp.values)))

    # Create DataFrame to hold index for each used gene.
    genes_with_index = pd.DataFrame({'gene_symbol': set_of_gene_symbols}) \
        .reset_index() \
        .rename(columns={'index': 'gene_index'})

    # Only take the neigbor tuples from the total neighborhood DataFrame which have both
    # genes in the set of genes of the node feature DataFrame's.
    neighbors_as_genes = [[neigh[0], neigh[1]] for neigh in gene_symbol_tuples \ 
        if ((neigh[0] in genes_with_index['gene_symbol'].values) & \ 
            (neigh[1] in genes_with_index['gene_symbol'].values))]

    # Transform genes to their corresponding indices.
    transform_gene_tuple_to_index_tuple = lambda x : [genes_with_index[genes_with_index.gene_symbol==x[0]]['gene_index'].values[0], 
                                                      genes_with_index[genes_with_index.gene_symbol==x[1]]['gene_index'].values[0]]
    neighbors_as_indices = [transform_gene_tuple_to_index_tuple(tup) for tup in neighbors_as_genes]

    # Set graph as value for the current cell line name.
    cell_line_graph[cln] = Data(x          = torch.tensor(node_features, 
                                                          dtype=torch.float),
                                edge_index = torch.tensor(neighbors_as_indices, 
                                                          dtype=torch.long))
