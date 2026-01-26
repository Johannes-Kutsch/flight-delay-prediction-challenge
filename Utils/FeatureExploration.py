import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from Utils.HelperFunctions import _aggregate_df
from Utils.Styling import RED_L, YEL_L, GRE_L, BLU_L, ORA, ORA_L, RES, PLOT_TITLE_FONTSIZE
from Utils.StringHelperFunctions import strip_ansi_codes
from typing import Any

def explore_features(df, feature_mask:list[str] = None, target_feature_name:str = None, target_feature: list[Any] = None,
                     plot_nan_heatmap: bool = True, print_nan_rows: bool = False, plot_duplicate_heatmap: bool = True):
    sns.set_style("whitegrid")

    df, has_target_feature, target_feature_name = _aggregate_df(df, feature_mask, target_feature, target_feature_name)

    num_rows = len(df)
    feature_descriptions = [['ID', 'Name', 'non-null count', 'D-Type', 'Feature Summary']]
    for i, col in enumerate(df.columns):
        feature_descriptions.append([i] + _create_feature_descriptions(df[col], num_rows))

    _print_feature_descriptions(feature_descriptions)

    if df.duplicated().any():
        extra_duplicates_count = df.duplicated().sum()
        total_rows = len(df)
        percent = extra_duplicates_count / total_rows * 100

        print(f"{ORA_L}DUPLICATE ENTRIES DETECTED: {extra_duplicates_count}/{total_rows} ({percent:.2f}%) additional duplicate entries{RES}\n")

        if plot_duplicate_heatmap:
            df['dup_cluster'] = df.groupby(list(df.columns)).ngroup()

            extra_dup_mask = df.duplicated(keep='first')
            heatmap_data = df['dup_cluster'].where(extra_dup_mask, -1).to_frame()

            colors = ['lightgrey'] + sns.color_palette('autumn', n_colors=20).as_hex()
            cmap = ListedColormap(colors)

            plt.figure(figsize=(5, 5.5))
            sns.heatmap(
                heatmap_data,
                cmap=cmap,
                cbar=False,
                linecolor='grey',
                yticklabels=False,
                xticklabels=False,
            )
            plt.title("Extra Duplicate Rows Highlighted by Cluster\n", fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold")
            plt.show()
            print()

    if plot_nan_heatmap and df.isnull().values.any():
        sns.heatmap(df.isnull(), cbar=False, cmap='bone_r')
        plt.title("non-null Distribution\n", fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold")
        plt.show()
        print()

    if print_nan_rows and df.isnull().values.any():
        print(f"{BLU_L}NaN Rows{RES}")
        display(df.loc[df.isna().any(axis=1)])


def _create_feature_descriptions(feature: pd.Series, data_count):
    dtype_str = _get_dtype_str(feature)
    non_null_str = _get_non_null_str(feature, data_count)
    feature_summary = _get_feature_summary(feature, data_count)

    return [feature.name, non_null_str, dtype_str, feature_summary]


def _get_dtype_str(feature):
    num_data_types = len(feature.dropna().map(type).unique())
    if num_data_types > 1:
        return f"{RED_L}multiple types{RES}"
    elif pd.api.types.is_object_dtype(feature):
        return f"{RED_L}{feature.dtype}{RES}"
    else:
        return f"{GRE_L}{feature.dtype}{RES}"

def _get_non_null_str(feature, data_count):
    n_non_null = feature.notna().sum()
    if data_count - n_non_null > 0:
        return f"{RED_L}{n_non_null} ({(1 - (n_non_null / data_count))*100:.2f}%){RES}"
    else:
        return f"{GRE_L}{n_non_null} non-null{RES}"

def _get_feature_summary(feature, data_count):
    num_unique = feature.nunique(dropna=True)
    unique_ratio = num_unique / data_count if data_count > 0 else 0

    if pd.api.types.is_bool_dtype(feature.dtype):
        counts = feature.value_counts(dropna=True)
        summary = f"{ORA}bool | "
        summary += ", ".join([f"{val} ({cnt})" for val, cnt in counts.items()])
        summary += RES
        return summary

    if (num_unique < 5 or unique_ratio < 0.05) and num_unique < 15:
        counts = feature.value_counts(dropna=True)
        summary = f"{BLU_L}{num_unique} unique | "
        if len(counts) == 0:
            summary += f"all items appear once"
        else:
            summary = f"{BLU_L}{num_unique} unique | "
            formatted_items = []
            for val, cnt in counts.items():
                if isinstance(val, (int, float, np.number)):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)
                formatted_items.append(f"{val_str} ({cnt})")
            summary += ", ".join(formatted_items)

        summary += RES
        return summary

    if pd.api.types.is_numeric_dtype(feature.dtype):
        summary = f"Min: {feature.min():.2f}, Max: {feature.max():.2f}, Median: {feature.median():.2f}, Mean: {feature.mean():.2f}"
        return summary
    else:
        counts = feature.value_counts(dropna=True).head(5)
        counts = counts[counts > 1]
        if len(counts) == 0:
            summary = f"{YEL_L}only unique Values{RES}"
        else:
            summary = ", ".join([f"{val} ({cnt})" for val, cnt in counts.items()])
        return summary


def _print_feature_descriptions(feature_descriptions):
    col_widths = [0] * len(feature_descriptions[0])

    for row in feature_descriptions:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(strip_ansi_codes(str(val))))

    for row in feature_descriptions:
        row_str = ""
        for i, val in enumerate(row):
            string_len = len(strip_ansi_codes(str(val)))
            row_str += str(val)
            row_str += " " * (col_widths[i] - string_len + 2)
        print(row_str)

    print()


def _ansi_format_left(string: str, width: int) -> str:
    visible_len = len(strip_ansi_codes(string))
    padding = max(width - visible_len, 0)
    return string + " " * padding
