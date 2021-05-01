from typing import Tuple, List, Any
import pandas as pd
from data_mining import ColorizedLogger

logger = ColorizedLogger('Preprocess', 'blue')


class Preprocess:
    __slots__ = ('sort_col', 'group_col')

    sort_col: str
    group_col: str

    def __init__(self, group_col: str, sort_col: str):
        self.sort_col = sort_col
        self.group_col = group_col

    def prepare_for_split(self, df: pd.DataFrame, num_to_trim: int = 7) -> pd.DataFrame:
        num_rows = df.shape[0]
        all_countries = df[self.group_col].unique()
        # Take the counts for the last 7 dates
        date_counts = df.groupby(self.sort_col).count().iso_code
        last_n_dates = date_counts.iloc[-num_to_trim:].index.tolist()
        # Get the count for n days ago and calculate its 90%
        count_n_days_ago = date_counts.iloc[-num_to_trim]
        least_value_to_keep = count_n_days_ago * 0.9
        # Drop rows from the last n dates with count less than this value
        cond = (df.date.isin(last_n_dates))
        df[cond] = df[cond].groupby(self.sort_col) \
            .filter(lambda x: x[self.group_col].count() > least_value_to_keep)
        df = df[df[self.group_col].notna()].reset_index(drop=True)
        # For the rows kept in these dates, find the countries missing
        # and drop all their corresponding rows
        date_counts = df.groupby(self.sort_col).count().iso_code
        last_n_dates = date_counts.iloc[-num_to_trim:].index.tolist()
        cond = (df.date.isin(last_n_dates))
        countries_present = df[cond].iso_code.unique()
        countries_missing = set(all_countries)-set(countries_present)
        df = df[~df.iso_code.isin(countries_missing)]

        rows_dropped = num_rows - df.shape[0]
        logger.info(f"{rows_dropped} rows dropped.")

        return df

    def expand_columns_on_country(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_expand = set(df.columns) - {self.group_col, self.sort_col}
        df = pd.pivot(df, index=self.sort_col, columns=self.group_col, values=cols_to_expand) \
            .reset_index()
        df.columns = df.columns.map(lambda col: col[0] if col[0] == 'date' else '$'.join(col))
        df = df.sort_index(axis='columns')
        df = df.fillna(0.0)
        return df

    def contract_columns_on_country(self, df: pd.DataFrame, target_columns: List[str]) \
            -> pd.DataFrame:

        for ind, base_col in enumerate(target_columns):
            df_part = df.copy()
            base_col_offsprings = [col for col in df_part.columns if base_col + '$' in col]
            df_part = df_part[base_col_offsprings + ['date']]

            df_part = df_part.melt(id_vars=['date'],
                                   value_vars=base_col_offsprings,
                                   var_name="iso_code",
                                   value_name=base_col)
            df_part.iso_code = df_part.iso_code.str.split('$').str[-1]
            df_part = df_part.sort_values(['date', 'iso_code']).reset_index(drop=True)

            if ind == 0:
                df_new = df_part.copy()
            else:
                non_common_cols = df_part.columns.difference(df_new.columns)
                for non_common_col in non_common_cols:
                    df_new[non_common_col] = df_part[non_common_col].to_numpy()

        return df_new
