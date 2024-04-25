import pandas as pd


def delete_rows_with_missing_values(df: pd.DataFrame):
    print(f"Number of rows before deleting: {df.shape[0]}")
    df.dropna()
    print(f"Number of rows after deleting: {df.shape[0]}")
