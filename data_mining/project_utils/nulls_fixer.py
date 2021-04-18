import numpy as np
import pandas as pd
from typing import List
from data_mining import ColorizedLogger

logger = ColorizedLogger('NullsFixer', 'white')


class NullsFixer:
    __slots__ = ('sort_col', 'group_col')

    sort_col: str
    group_col: str
    cols: List[str] = ['iso_code', 'date', 'daily_vaccinations', 'total_vaccinations',
                       'people_vaccinated', 'people_fully_vaccinated']

    def __init__(self, sort_col: str, group_col: str):
        self.sort_col = sort_col
        self.group_col = group_col

    def fix_and_infer(self, df: pd.DataFrame) -> pd.DataFrame:
        accum_cols = ['people_fully_vaccinated', 'people_vaccinated', 'total_vaccinations']
        df = self.fix(df)
        for col in accum_cols:
            count_nan = len(df[col]) - df[col].count()
            if count_nan > 0:
                df = self.infer_accum_col(df, col)
            df = self.fix(df)

        return df

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        all_cols = df.columns
        nulls_prev = df.loc[:, self.cols].isna().sum()
        while True:
            df = self.fix_people_fully_vaccinated(df)
            df = self.fix_people_vaccinated(df)
            df = self.fix_total_vaccinations(df)
            df = self.fix_daily_vaccinations(df)
            nulls = df.loc[:, self.cols].isna().sum()
            if nulls.equals(nulls_prev):
                break
            nulls_prev = nulls

        return df.loc[:, all_cols]

    def infer_accum_col(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        def _infer_values(col, col_list, nulls_idx, val, consecutive_nulls):
            # Get top and bottom non-null values (for this block of consecutive nulls)
            non_null_val_1 = col[col_list[nulls_idx[0] - 1][0]]
            non_null_val_2 = val
            # Calculate avg difference and create whole-number steps
            diff = non_null_val_2 - non_null_val_1
            whole_step, remainder = divmod(diff, consecutive_nulls + 1)
            steps = whole_step * np.ones(consecutive_nulls)
            steps[1:int(remainder) + 1] += 1
            # Add the avg steps to each null value for this block
            for null_ind, step in zip(nulls_idx, steps):
                pd_idx_previous = col_list[null_ind - 1][0]
                val_to_insert = col[pd_idx_previous] + step
                pd_idx_null_current = col_list[null_ind][0]
                col[pd_idx_null_current] = val_to_insert

            return col

        def f_cols(col):
            consecutive_nulls = 0
            nulls_idx = []
            col_list = [(idx, val) for idx, val in col.items()]
            for ind, (pd_ind, val) in enumerate(col_list):
                if pd.isna(val):
                    if ind == 0:
                        col[pd_ind] = 0.0
                    else:
                        consecutive_nulls += 1
                        nulls_idx.append(ind)
                        if ind == len(col_list) - 1:
                            non_null_val_1 = col[col_list[nulls_idx[0] - 1][0]]
                            mean_step = round(col.mean())
                            max_val = non_null_val_1 + mean_step * consecutive_nulls
                            col = _infer_values(col, col_list, nulls_idx, max_val, consecutive_nulls)
                else:
                    if consecutive_nulls > 0:
                        col = _infer_values(col, col_list, nulls_idx, val, consecutive_nulls)
                        # Reset
                        consecutive_nulls = 0
                        nulls_idx = []

            return col

        def f_groups(df: pd.DataFrame, col: str):
            df.loc[:, [col]] = df[[col]].apply(f_cols, axis=0)
            return df

        df = df.groupby(df[self.group_col]).apply(f_groups, col)

        return df

    @classmethod
    def fix_people_fully_vaccinated(cls, df: pd.DataFrame) -> pd.DataFrame:
        def f1(row):
            cond_1 = pd.notna(row['total_vaccinations']) and pd.notna(row['people_vaccinated'])
            cond_2 = pd.isna(row['people_fully_vaccinated'])
            if cond_1 and cond_2:
                row = row['total_vaccinations'] - row['people_vaccinated']
            else:
                row = row['people_fully_vaccinated']
            return row

        def f2(row):
            cond_1 = row['total_vaccinations'] == 0.0
            cond_2 = pd.isna(row['people_fully_vaccinated'])
            if cond_1 and cond_2:
                row = 0.0
            else:
                row = row['people_fully_vaccinated']
            return row

        # people_fully_vaccinated = total_vaccinations - people_vaccinated
        df.loc[:, 'people_fully_vaccinated'] = df.apply(f1, axis=1)
        # If total_vaccinations==0 -> people_fully_vaccinated = 0.0
        df.loc[:, 'people_fully_vaccinated'] = df.apply(f2, axis=1)
        # if prev_col == next_col -> col=prev_col
        cls.fix_if_unchanged(df, 'people_fully_vaccinated')

        return df

    @classmethod
    def fix_people_vaccinated(cls, df: pd.DataFrame) -> pd.DataFrame:
        def f1(row):
            cond_1 = pd.notna(row['total_vaccinations']) and pd.notna(row['people_fully_vaccinated'])
            cond_2 = pd.isna(row['people_vaccinated'])
            if cond_1 and cond_2:
                row = row['total_vaccinations'] - row['people_fully_vaccinated']
            else:
                row = row['people_vaccinated']
            return row

        def f2(row):
            cond_1 = row['total_vaccinations'] == 0.0
            cond_2 = pd.isna(row['people_vaccinated'])
            if cond_1 and cond_2:
                row = 0.0
            else:
                row = row['people_vaccinated']
            return row

        # people_vaccinated = total_vaccinations - people_fully_vaccinated
        df.loc[:, 'people_vaccinated'] = df.apply(f1, axis=1)
        # If total_vaccinations==0 -> people_vaccinated = 0.0
        df.loc[:, 'people_vaccinated'] = df.apply(f2, axis=1)
        # if prev_col == next_col -> col=prev_col
        cls.fix_if_unchanged(df, 'people_vaccinated')

        return df

    @classmethod
    def fix_total_vaccinations(cls, df: pd.DataFrame) -> pd.DataFrame:
        def f1(row):
            cond_1 = pd.notna(row['people_vaccinated']) and pd.notna(row['people_fully_vaccinated'])
            cond_2 = pd.isna(row['total_vaccinations'])
            if cond_1 and cond_2:
                row = row['people_vaccinated'] + row['people_fully_vaccinated']
            else:
                row = row['total_vaccinations']
            return row

        def f2(row):
            cond_1 = pd.notna(row['previous_total_vaccinations']) and pd.notna(
                row['daily_vaccinations'])
            cond_2 = pd.isna(row['total_vaccinations'])
            if cond_1 and cond_2:
                row = row['previous_total_vaccinations'] + row['daily_vaccinations']
            else:
                row = row['total_vaccinations']
            return row

        def f3(row):
            cond_1 = pd.notna(row['next_total_vaccinations']) and \
                     pd.notna(row['next_daily_vaccinations'])
            cond_2 = pd.isna(row['total_vaccinations'])
            if cond_1 and cond_2:
                row = row['next_total_vaccinations'] - row['next_daily_vaccinations']
            else:
                row = row['total_vaccinations']
            return row

        # total_vaccinations = people_vaccinated + people_fully_vaccinated
        df.loc[:, 'total_vaccinations'] = df.apply(f1, axis=1)
        # total_vaccinations = previous_total_vaccinations + daily_vaccinations
        df['previous_total_vaccinations'] = \
            df['total_vaccinations'].groupby(df['iso_code']).shift(1, fill_value=0.0)
        df.loc[:, 'total_vaccinations'] = df.apply(f2, axis=1)
        # total_vaccinations = next_total_vaccinations - next_daily_vaccinations
        df['next_total_vaccinations'] = df['total_vaccinations'].groupby(df['iso_code']).shift(-1)
        df['next_daily_vaccinations'] = df['daily_vaccinations'].groupby(df['iso_code']).shift(-1)
        df.loc[:, 'total_vaccinations'] = df.apply(f3, axis=1)
        # if prev_col == next_col -> col=prev_col
        cls.fix_if_unchanged(df, 'total_vaccinations')

        return df

    @classmethod
    def fix_daily_vaccinations(cls, df: pd.DataFrame) -> pd.DataFrame:
        def f1(row):
            cond_1 = pd.notna(row['total_vaccinations']) and \
                     pd.notna(row['previous_total_vaccinations'])
            cond_2 = pd.isna(row['daily_vaccinations'])
            if cond_1 and cond_2:
                row = row['total_vaccinations'] - row['previous_total_vaccinations']
            else:
                row = row['daily_vaccinations']
            return row

        # daily_vaccinations = total_vaccinations - previous_total_vaccinations
        df['previous_total_vaccinations'] = \
            df['total_vaccinations'].groupby(df['iso_code']).shift(1, fill_value=0.0)
        df.loc[:, 'daily_vaccinations'] = df.apply(f1, axis=1)
        # if prev_col == next_col -> col=prev_col
        cls.fix_if_unchanged(df, 'daily_vaccinations')

        return df

    @staticmethod
    def fix_if_unchanged(df: pd.DataFrame, col: str) -> pd.DataFrame:
        def f1(row):
            cond_1 = pd.notna(row[f'previous_{col}']) and pd.notna(row[f'next_{col}'])
            cond_2 = row[f'previous_{col}'] == row[f'next_{col}']
            cond_3 = pd.isna(row[col])
            if cond_1 and cond_2 and cond_3:
                row = row[f'previous_{col}']
            else:
                row = row[col]
            return row

        # if prev_col == next_col -> col=prev_col
        df[f'previous_{col}'] = df[col].groupby(df['iso_code']).shift(1, fill_value=0.0).ffill(axis=0)
        df[f'next_{col}'] = df[col].groupby(df['iso_code']).shift(-1).bfill(axis=0)
        df.loc[:, col] = df.apply(f1, axis=1)

        return df
