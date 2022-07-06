import pandas as pd

from typing import List


def map_gene_to_number(nodes_as_numbers: List[int], nodes_as_genes: List[str]): 
    return dict(zip(nodes_as_numbers, nodes_as_genes))

def map_number_to_gene(nodes_as_numbers: List[int], nodes_as_genes: List[str]): 
    return dict(zip(nodes_as_genes, nodes_as_numbers))

def map_gene_symbols_to_proteins(
    proteins: pd.DataFrame, 
    protein_genes: pd.DataFrame, 
    left_on: str,
    new_col_name: str):
    """Maps the corresponding gene symbols to each protein in the given dataframe.

    Args:
        proteins (pd.DataFrame): 
            Dataframe containing a column with the a protein identifier.
        protein_genes (pd.DataFrame):
            Dataframe containing a columns for a protein identifier and the
            corresponding gene symbol.
        left_on (str):
            Column of the left table to join the protein identifier of the right table 
            on.
        new_col_name (str): 
            Name of the new gene symbol column.

    Returns: 
        pd.DataFrame
            New dataframe containing the 'proteins' dataframe with their mapped gene 
            symbols.
    """
    return proteins.merge(right    = protein_genes[['string_protein_id', 'preferred_name']],
                          how      = 'left',
                          left_on  = left_on,
                          right_on = 'string_protein_id') \
                    .rename(columns={'preferred_name': new_col_name}, inplace=True) \
                    .drop(['string_protein_id'], axis=1, inplace=True)
