import pandas as pd
from typing import List


def convert_column_types(
    df: pd.DataFrame,
    cols_to_convert: List[str],
    conversion_type: str = 'float64'
    ) -> pd.DataFrame:
    """
    Converts given columns of a given pd.DataFrame to a specific type.

    Arguments:
        df : pd.DataFrame
            Input dataframe in which the type of a set of columns will be converted.
        cols_to_convert : List[str]
            The set of columns which will be converted.
        conversion_type : str, default = 'float64'
            The type in which the specified columns will be converted to.

    Returns:
        pd.DataFrame
            Input dataframe with the specified columns converted to the wanted type.
    """
    # Convert the wanted columns to the specified type.
    converted_subset = df[cols_to_convert].astype(conversion_type)
    assert all(converted_subset.dtypes == conversion_type),\
        f"ERROR: Some columns were not converted to type `{conversion_type}`."

    # The initial pd.DataFrame without the wanted columns.    
    df_subset = df.loc[:, ~df.columns.isin(cols_to_convert)]
    assert df_subset.shape[0] == df.shape[0],\
        f"ERROR: Number of rows differ because subset = {df_subset.shape[0]} != {df.shape[0]} = initial."
    assert df_subset.shape[1] == df.shape[1] - len(cols_to_convert),\
        f"ERROR: Number of cols differ because subset = {df_subset.shape[1]} != {df.shape[1] - len(cols_to_convert)} = initial."

    # Add the converted columns as new columns in the subset.
    df_converted = pd.concat([df_subset, converted_subset], axis=1) # df_converted = converted_subset.copy()
    assert df_converted.shape == df.shape,\
        f"ERROR: New shape differs because New = {df_converted.shape} != {df.shape} = initial." 

    return df_converted