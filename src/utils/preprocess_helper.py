from typing import Optional, Set
from pandas._libs.parsers import ParserError
import io 
import pandas as pd


def get_gdsc_gene_expression(
    genes: Optional[Set] = None,
    path_cell_annotations: str = "data/GDSC/Cell_Lines_Details.csv",
    path_gene_expression: str = "data/GDSC/Cell_line_RMA_proc_basalExp.txt",
):
    """
    Returns the gene expression dataframe(n_cells x n_genes) for a 
    set of gene symbols for all cell_lines of the GDSC cell line annotation file.
    If the genes are None, return the data for all genes.
    """

    gene_expression = pd.read_csv(path_gene_expression, sep="\t")
    gene_expression = gene_expression.rename(columns={'GENE_SYMBOLS': 'Sample Name'}, level=0)
    gene_expression = gene_expression\
        .drop(["GENE_title"], axis=1)\
        .set_index("Sample Name")
    gene_expression.index = gene_expression.index.astype(str)

    # Refactor column names to cosmic id and then map to cell-line name.
    ge_columns = [
        x.split("DATA.")[1] for x in list(gene_expression.columns)
    ] # Remove "DATA" prefix.
    ge_columns = cosmic_ids_to_cell_line_names(
        ge_columns, path_cell_annotations=path_cell_annotations
    )
    gene_expression.columns = ge_columns.astype(str)

    if genes is None:
        return gene_expression.T
    else:
        # Filter out the genes.
        number_of_queried_genes = len(genes)
        genes = set(gene_expression.index) & genes
        print(
            f"No data for {number_of_queried_genes - len(genes)} of {number_of_queried_genes} queried genes."
        )
        gene_expression = gene_expression.loc[genes]

    return gene_expression.T


def cosmic_ids_to_cell_line_names(
    cosmic_ids, 
    path_cell_annotations="data/GDSC/Cell_Lines_Details.csv"
):
    """
    Transforms a list of COSMIC ID's to a series of cell-line-names, indexed by the cosmic ID
    using the cell annotations from https://www.cancerrxgene.org/downloads/bulk_download.
    """

    try:
        if path_cell_annotations[-4:] == '.csv': 
            cell_line_data = pd.read_csv(path_cell_annotations, index_col=0)
        elif path_cell_annotations[-5:] == '.xlsx':
            cell_line_data = pd.read_excel(path_cell_annotations)
    except ParserError:
        csv_data = open(path_cell_annotations).read().replace("\r\n", "\n")
        cell_line_data = pd.read_csv(io.StringIO(csv_data), encoding="unicode_escape")

    cosmic_ids_to_cell_line_name_dict = pd.Series(
        cell_line_data["Sample Name"].values,
        index=cell_line_data["COSMIC identifier"].fillna(-1).astype(int).values,
    ).to_dict()

    cell_line_names = []
    unknown_cell_line_names = []
    for cosmic_id in cosmic_ids:
        try:
            cell_line_names.append(cosmic_ids_to_cell_line_name_dict[int(cosmic_id)])
        except (KeyError, ValueError):
            cell_line_names.append("unknown_cosmic_" + str(cosmic_id))
            unknown_cell_line_names.append(cosmic_id)

    if unknown_cell_line_names:
        print(
            "Note: "
            + str(len(unknown_cell_line_names))
            + " Cosmic IDs not found in cell annotation data: "
        )
        print(unknown_cell_line_names)

    # Check if cell_line_names are unique.
    unique_c = []
    dup_c = []
    for c in cell_line_names:
        if not (c in unique_c):
            unique_c.append(c)
        else:
            dup_c.append(c)
    if dup_c:
        print(
            "Warning: at least two cosmic IDs map to the same cell lines for the cell lines: "
        )
        print(dup_c)

    return pd.Series(cell_line_names, index=cosmic_ids)