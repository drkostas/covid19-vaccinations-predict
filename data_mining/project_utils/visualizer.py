import warnings

import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.dates import DateFormatter
from typing import List

from data_mining import ColorizedLogger

warnings.filterwarnings("ignore")
rcParams['axes.titlepad'] = 20
pd.set_option('display.max_rows', 500)
logger = ColorizedLogger('Visualizer', 'cyan')


class Visualizer:
    __slots__ = ('sort_col', 'group_col', 'text_color')

    sort_col: str
    group_col: str
    text_color: str

    def __init__(self, sort_col: str, group_col: str, text_color: str):
        self.sort_col = sort_col
        self.group_col = group_col
        self.text_color = text_color

    @staticmethod
    def plot_categorical_val_counts(df: pd.DataFrame, cols_to_visualize: List[str],
                                    print_values: bool = False, top: int = 10) -> None:
        # Copy the DF
        df_ = df.copy()

        # Calculate Value Counts
        val_counts = {}
        for col in cols_to_visualize:
            val_counts[col] = df_[col].value_counts().nlargest(top)
            if print_values:
                logger.info(f"Column {col.capitalize()} Value Counts:\n{val_counts[col]}\n")
        # Plot them
        sns.set(font_scale=1.0)
        fig, ax = plt.subplots(nrows=1, ncols=len(cols_to_visualize), figsize=(15, 10))
        for ind, cat_col in enumerate(cols_to_visualize):
            ax[ind].set_title(f"Top {top} {cat_col} value counts", fontsize=18)
            ax[ind].set_xticklabels(ax[ind].get_xticklabels(),
                                    rotation=90, ha="right", fontsize=14)
            ax[ind].set_ylabel(ax[ind].get_ylabel(), fontsize=16)
            val_counts[cat_col].index = val_counts[cat_col].index.str.slice(stop=40)
            sns.barplot(x=val_counts[cat_col].index,
                        y=val_counts[cat_col], ax=ax[ind])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_vaccines_val_counts(df: pd.DataFrame, cols_to_visualize: List[str],
                                 print_values: bool = False, top: int = 10) -> None:
        # Copy the DF
        df_ = df.copy()

        # Calculate Value Counts
        val_counts = {}
        for col in cols_to_visualize:
            val_counts[col] = int(df_[col].sum())
            if print_values:
                logger.info(f"{col}: {val_counts[col]}")

        # Plot them
        sns.set(font_scale=1.0)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
        val_counts = {k: v for k, v in sorted(val_counts.items(),
                                              key=lambda item: item[1],
                                              reverse=True)}
        ax = sns.barplot(x=pd.Series(val_counts.keys()),
                         y=pd.Series(val_counts.values()), ax=ax)
        ax.set_title(f"Number of day entries for each vaccine", fontsize=28)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=20)
        ax.set_yticklabels(ax.get_yticks(), fontsize=20)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def viz_columns_corr(df: pd.DataFrame, cols_to_visualize: List[str]) -> None:
        sns.set(font_scale=1.4)
        sns.heatmap(data=df[cols_to_visualize].corr(), cmap='coolwarm', annot=True, fmt=".1f",
                    annot_kws={'size': 16})

    def plot_numerical_val_counts_per_country(self, df: pd.DataFrame, top: int = 3) -> None:
        # Copy the DF
        df_ = df.copy()

        val_counts = df_[self.group_col].value_counts().nlargest(top).index
        for iso_code in val_counts:
            df_filtered = df_[df_[self.group_col] == iso_code]
            # Configure the grid of the plot
            fig, _ = plt.subplots(nrows=5, ncols=1, figsize=(15, 20))
            ax = [plt.subplot2grid((3, 2), (0, 0), fig=fig), plt.subplot2grid((3, 2), (0, 1), fig=fig),
                  plt.subplot2grid((3, 2), (1, 0), fig=fig), plt.subplot2grid((3, 2), (1, 1), fig=fig),
                  plt.subplot2grid((3, 2), (2, 0), fig=fig), plt.subplot2grid((3, 2), (2, 1), fig=fig)]
            # plt.subplots_adjust(hspace=0.6)

            # Plot using Pandas
            # First Row - First Column
            ax[0].set_title(f'Total Vaccinations (logscale) [{iso_code}]', fontsize=20,
                            c=self.text_color)
            df_filtered[['people_vaccinated', 'people_fully_vaccinated']].plot.hist(ax=ax[0],
                                                                                    fontsize=11,
                                                                                    alpha=0.7,
                                                                                    logy=True,
                                                                                    stacked=True)
            ax[0].set_ylabel('# of Daily Entries', fontsize=16, c=self.text_color)
            # First Row - Second Column
            df_filtered[['total_vaccinations']] \
                .plot.hist(ax=ax[1], fontsize=11, alpha=0.7, logy=True)
            ax[1].set_ylabel('')
            # Second Row - First Column
            ax[2].set_title(f'Vaccinations per hundred (logscale) [{iso_code}]', fontsize=20,
                            c=self.text_color)
            df_filtered[
                ['people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred']].plot.hist(
                ax=ax[2], fontsize=15, alpha=0.7, logy=True, stacked=True)
            ax[2].set_ylabel('# of Daily Entries', fontsize=16, c=self.text_color)
            # Second Row - Second Column
            df_filtered[['total_vaccinations_per_hundred']].plot.hist(ax=ax[3], fontsize=15, alpha=0.7,
                                                                      logy=True)
            ax[3].set_ylabel('', fontsize=16, c=self.text_color)
            # Third Row - First Column
            ax[4].set_title(f'Daily Vaccinations (logscale) [{iso_code}]', fontsize=20,
                            c=self.text_color)
            df_filtered[['daily_vaccinations']].plot.hist(ax=ax[4], fontsize=15, alpha=0.7, logy=True,
                                                          stacked=True)
            ax[4].set_ylabel('# of Daily Entries', fontsize=16, c=self.text_color)
            # Third Row - Second Column
            df_filtered[['daily_vaccinations_per_million']].plot.hist(ax=ax[5], fontsize=15, alpha=0.7,
                                                                      logy=True)
            ax[5].set_ylabel('', fontsize=16, c=self.text_color)
            # Last configurations
            for subax in ax:
                subax.set_xlabel('Frequency', fontsize=16, c=self.text_color)
                subax.tick_params(axis="both", colors=self.text_color)
                subax.legend(loc='upper right', prop={'size': 14})
            # Show plots
            plt.tight_layout()
            plt.show()

    @staticmethod
    def viz_missing_values(df: pd.DataFrame, cols: List[str], skip_1: bool = False,
                           skip_2: bool = False, print_values: bool = False) -> None:
        # Copy the DF
        df_ = df.copy()

        if print_values:
            logger.info(f"Missing Values:\n{df[cols].isna().sum()}")
        df_ = df_[cols]
        # msno.bar(covid_df, fontsize=24, color="dodgerblue")
        # Visualize rows to find which columns are null
        if not skip_1:
            msno.matrix(df_, fontsize=28, color=(0.118, 0.565, 1))  # , sort='ascending')
        # Visualize heatmap to find correlations between columns when they have null values
        if not skip_2:
            msno.heatmap(df_, cmap="RdBu", fontsize=28)

    def viz_top_countries_accumulated_statistics(self, df: pd.DataFrame, top_n: int = 10) -> None:
        # Copy the DF
        df_ = df.copy()

        # Plot total vaccinations per country (top 25)
        vacc_amount = df_.groupby(self.group_col).max() \
            .sort_values('total_vaccinations', ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(12, 25), nrows=3, ncols=1)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.6)
        ax[0].bar(vacc_amount.index, vacc_amount.total_vaccinations)
        ax[0].tick_params(labelrotation=90, axis='x')
        ax[0].set_title(f"Total vaccinations (Top {top_n} Countries)")
        ax[0].set_ylabel('Number of vaccinated citizens')
        ax[0].set_xlabel('Countries')
        # Plot total vaccinations per 100 per country (top 25)
        vacc_amount = df_.groupby(self.group_col).max() \
            .sort_values('total_vaccinations_per_hundred', ascending=False).head(top_n)
        ax[1].bar(vacc_amount.index, vacc_amount.total_vaccinations_per_hundred)
        ax[1].tick_params(labelrotation=90, axis='x')
        ax[1].set_title(f"Total vaccinations per 100 (Top {top_n} Countries)")
        ax[1].set_ylabel('Number of vaccinated citizens per 100 citizens')
        ax[1].set_xlabel('Countries')
        # AVG daily vaccinations per 100 per country(top 25)
        vacc_amount = df_.groupby(self.group_col).mean() \
            .sort_values('daily_vaccinations_per_hundred', ascending=False).head(top_n)
        ax[2].bar(vacc_amount.index, vacc_amount.daily_vaccinations_per_hundred)
        ax[2].tick_params(labelrotation=90, axis='x')
        ax[2].set_title(f"AVG daily vaccinations per 100 (Top {top_n} Countries)")
        ax[2].set_ylabel('AVG daily vaccinations per 100')
        ax[2].set_xlabel('Countries')
        # Show Plot
        plt.tight_layout()
        plt.show()

    def viz_top_countries_progress(self, df: pd.DataFrame, top_sort_by: str, top_n: int = 10) -> None:
        # Copy the DF
        df_ = df.copy().sort_values(self.sort_col)
        # Initialize the figure
        plot_cols = ['total_vaccinations_per_hundred',
                     'people_vaccinated_per_hundred',
                     'people_fully_vaccinated_per_hundred',
                     'daily_vaccinations_per_hundred']
        fig, ax = plt.subplots(figsize=(16, 32), nrows=4, ncols=1)
        # Prepare the Data
        top_countries = list(df_.groupby(self.group_col).max()
                             .sort_values(top_sort_by, ascending=False)
                             .head(top_n).index)
        # Setup the figure
        for ax_, plot_col in zip(ax, plot_cols):
            df_groups = df_.loc[df_[self.group_col].isin(top_countries),
                                [self.group_col, self.sort_col, plot_col]] \
                .groupby(df_[self.group_col])
            for name, group_df in df_groups:
                ax_ = group_df.plot.line(x=self.sort_col, y=plot_col, ax=ax_, label=name,
                                         legend=True, subplots=False)
                ax_.set_title(f"{plot_col.capitalize()} Progression", fontsize=24)
                ax_.set_xlabel('Date', fontsize=22)
                ax_.set_ylabel('% Value', fontsize=22)
                ax_.tick_params(axis='both', size=20)
                ax_.legend(prop={'size': 16})
                plt.tight_layout()
                plt.subplots_adjust(hspace=0.6)
        # Show Plot
        plt.show()

    def viz_evaluate_predictions_progress(self, df_true: pd.DataFrame, df_pred: pd.DataFrame,
                                          plot_col: str, top_n: int = 10) -> None:
        # Copy the DF
        df_true_ = df_true.copy()
        df_pred_ = df_pred.copy()
        df_true_ = df_true_.sort_values(self.sort_col)  # .set_index(df_true_[self.sort_col])
        df_pred_ = df_pred_.sort_values(self.sort_col)  # .set_index(df_pred_[self.sort_col])
        # Initialize the figure
        plot_cols = [plot_col]
        fig, ax = plt.subplots(figsize=(12, 8 * top_n), nrows=top_n, ncols=len(plot_cols))
        date_form = DateFormatter("%m-%d")
        ax = np.array(ax).T
        # Prepare the Data
        dates = df_true_.date.unique()[-30:]
        top_countries = list(df_true_[df_true_.date.isin(dates)].groupby(self.group_col).max()
                             .sort_values(plot_col, ascending=False)
                             .head(top_n).index)
        # top_countries = ['USA', 'CHN']
        # Setup the figure
        if len(plot_cols) == 1:
            ax = [ax]
        for ax_group, plot_col in zip(ax, plot_cols):
            df_true_groups = df_true_.loc[df_true_[self.group_col].isin(top_countries),
                                          [self.group_col, self.sort_col, plot_col]].groupby(
                df_true_[self.group_col])
            df_pred_groups = df_pred_.loc[df_pred_[self.group_col].isin(top_countries),
                                          [self.group_col, self.sort_col, plot_col]].groupby(
                df_true_[self.group_col])
            if top_n == 1:
                ax_group = [ax_group]
            iter_groups = zip(df_true_groups, df_pred_groups, ax_group)
            for (name_true, true_group), (name_pred, pred_group), ax_ in iter_groups:
                true_max = float(true_group[plot_col].max())
                ax_ = true_group.plot.line(x=self.sort_col, y=plot_col, ax=ax_,
                                           label='true', legend=True, subplots=False)
                ax_ = pred_group.plot.line(x=self.sort_col, y=plot_col, ax=ax_,
                                           label='predictions', legend=True, subplots=False)
                plot_col_name = plot_col.replace('_', ' ').capitalize()
                ax_.set_title(f"{plot_col_name} progress for {name_true}",
                              fontsize=24, color=self.text_color)
                ax_.set_xlabel('Date', fontsize=22, color=self.text_color)
                ax_.set_ylabel('Value %', fontsize=22, color=self.text_color)
                ax_.set_ylim((-0.5, 4 * true_max))
                ax_.set_xticklabels(true_group.date, size=20,
                                    color=self.text_color, minor=False)
                ax_.xaxis.set_major_formatter(date_form)
                ax_.set_yticklabels(np.around(ax_.get_yticks(), decimals=4),
                                    size=20, color=self.text_color)
                ax_.legend(prop={'size': 16})
                # ax_.tick_params(which='major', size=20, color=self.text_color)
                ax_.tick_params(axis='both', which='minor', bottom=False, labelbottom=False)
        # Show Plot
        plt.minorticks_off()
        plt.tight_layout()
        # plt.subplots_adjust(hspace=0.6)
        plt.show()

    def viz_erros(self, per_date_avg: pd.Series, per_country_avg: pd.Series, top_n: int = 10) -> None:
        # Per country
        fig, ax = plt.subplots(figsize=(24, 8), nrows=1, ncols=2)
        # Prepare the Data
        countries_largest_errors = per_country_avg.sort_values(ascending=False).head(top_n)
        countries_smallest_errors = per_country_avg.sort_values(ascending=False).tail(top_n)
        # Setup the figure
        # Countries with largest error
        ax[0] = sns.barplot(x=countries_largest_errors.index,
                            y=countries_largest_errors, ax=ax[0])
        ax[0].set_title(f"Countries with largest errors", fontsize=24)
        ax[0].set_xlabel('Country', fontsize=22)
        ax[0].set_ylabel('RMSE', fontsize=22)
        ax[0].tick_params(axis='both', size=20)
        # Countries with smallest error
        ax[1] = sns.barplot(x=countries_smallest_errors.index,
                            y=countries_smallest_errors, ax=ax[1])
        ax[1].set_title(f"Countries with smallest errors", fontsize=24)
        ax[1].set_xlabel('Country', fontsize=22)
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].tick_params(axis='both', size=20)

        # Per Date
        fig, ax = plt.subplots(figsize=(24, 8), nrows=1, ncols=2)
        # Prepare the Data
        dates_largest_errors = per_date_avg.sort_values(ascending=False).head(top_n)
        dates_smallest_errors = per_date_avg.sort_values(ascending=False).tail(top_n)
        idx_large = dates_largest_errors.index.strftime("%d/%m")
        idx_small = dates_smallest_errors.index.strftime("%d/%m")
        # Setup the figure
        # Countries with largest error
        ax[0] = sns.barplot(x=idx_large,
                            y=dates_largest_errors, ax=ax[0])
        ax[0].set_title(f"Dates with largest errors", fontsize=24)
        ax[0].set_xlabel('Country', fontsize=22)
        ax[0].set_ylabel('RMSE', fontsize=22)
        ax[0].tick_params(axis='both', size=20)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")
        # Countries with smallest error
        ax[1] = sns.barplot(x=idx_small,
                            y=dates_smallest_errors, ax=ax[1])
        ax[1].set_title(f"Dates with smallest errors", fontsize=24)
        ax[1].set_xlabel('Date', fontsize=22)
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].tick_params(axis='both', size=20)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right")
        # Show Plot
        plt.tight_layout()
        # plt.subplots_adjust(hspace=0.6)
        plt.show()
