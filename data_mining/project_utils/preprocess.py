import numpy as np
import pandas as pd
from typing import List, Tuple
from data_mining import ColorizedLogger

logger = ColorizedLogger('Preprocess', 'blue')


class Preprocess:
    __slots__ = ('sort_col', 'group_col')

    sort_col: str
    group_col: str

    def __init__(self, group_col: str, sort_col: str):
        self.sort_col = sort_col
        self.group_col = group_col

    def pivot_columns_on_country(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_expand = set(df.columns) - {self.group_col, self.sort_col}
        df = pd.pivot(df, index=self.sort_col,
                      columns=self.group_col, values=cols_to_expand).reset_index()
        new_columns = []
        for col, country in df.columns:
            if col in cols_to_expand:
                new_columns.append(f"{col}_{country}")
            else:
                new_columns.append(col)
        df.columns = new_columns
        df = df.sort_index(axis='columns')

        return df

