import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Any

from pandas import DataFrame

from Utils.HelperFunctions import _aggregate_df
from Utils.Styling import PLOT_COLORS, PLOT_TITLE_FONTSIZE, TITLE_FONTSIZE


def explore_dataframe(df: pd.DataFrame, target_feature: list[Any] = None, target_feature_name:str = None, feature_mask:list[str] = None,
                      numerical_feature_names: list[str] = None, categorical_feature_names :list[str] = None, plot_data_spread = True,
                      plot_corr_matrix : bool = True, plot_pairs :bool = True, plot_target_correlation : bool = True):
    sns.set_style("whitegrid")
    df, has_target_feature, target_feature_name = _aggregate_df(df, feature_mask, target_feature, target_feature_name)

    categorical_features, continues_features = _sort_features(categorical_feature_names, df, numerical_feature_names)

    has_continues_features = len(continues_features) > 0
    has_categorical_features = len(categorical_features) > 0

    if plot_data_spread and has_continues_features:
        _plot_continues_features(df, continues_features)

    if plot_data_spread and has_categorical_features:
        _plot_categorical_features(df, categorical_features)

    if plot_pairs and has_target_feature:
        _plot_pairs(df, categorical_features, continues_features, target_feature_name)

    if plot_corr_matrix and len(categorical_features) > 1:
        _plot_correlation_matrix(df)

    if plot_target_correlation and has_target_feature:
        _plot_target_correlation(df, target_feature_name, categorical_features + continues_features)


def _sort_features(categorical_feature_names: list[str] | None, df, numerical_feature_names: list[str] | None) -> tuple[
    list[Any], list[Any]]:
    categorical_features = []
    continues_features = []

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        elif numerical_feature_names and numerical_feature_names.__contains__(col):
            continues_features.append(col)
        elif categorical_feature_names and categorical_feature_names.__contains__(col):
            categorical_features.append(col)
        else:
            num_unique = df[col].nunique()
            unique_ratio = num_unique / len(df)

            if num_unique < 5 or unique_ratio < 0.05 and num_unique < 15:
                categorical_features.append(col)
            else:
                continues_features.append(col)
    return categorical_features, continues_features

def _plot_pairs(df: pd.DataFrame, categorical_features: list, continues_features: list,
                target_feature_name: str):
    analyze_cols = categorical_features + continues_features
    analyze_cols = [c for c in analyze_cols if c != target_feature_name]

    df = pd.concat(
        [df[analyze_cols], df[target_feature_name]],
        axis=1
    )
    df = df.dropna()

    is_target_continues = continues_features.__contains__(target_feature_name)
    plot = None

    if is_target_continues:
        if is_target_continues:
            plot = sns.PairGrid(df, vars=analyze_cols, corner=True)
            plot.map_lower(lambda x, y, **kwargs: plt.scatter(x, y, c=df[target_feature_name], cmap="Blues", s=40, alpha=0.6))
            plot.map_diag(sns.kdeplot, fill=True)

            norm = plt.Normalize(df[target_feature_name].min(), df[target_feature_name].max())
            sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
            sm.set_array([])
            plot.figure.colorbar(sm, label=target_feature_name)

    else:
        plot = sns.pairplot(
            df,
            hue=target_feature_name,
            corner=True,
            diag_kind="kde",
            palette=PLOT_COLORS,
            plot_kws={"alpha": 0.6},
            height=3.2,
            aspect=1.1
        )

        if plot.legend is not None:
            plot.legend.set_title(f"{target_feature_name}\n", prop={"size": PLOT_TITLE_FONTSIZE, "weight": "bold"})
            for text in plot.legend.texts:
                text.set_fontsize(24)

    for ax in plot.axes.flatten():
        if ax is not None:
            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)
            ax.xaxis.label.set_weight("bold")
            ax.yaxis.label.set_weight("bold")
            ax.tick_params(labelsize=16)

    plot.figure.suptitle(
        "Pairwise Feature–Target Relationships",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
        y=1.02
    )

    plt.show()


def _plot_categorical_features(df: DataFrame, features: list[Any], n_cols: int = 2):
    n_rows = (len(features) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    total_n = len(df)
    i = None

    for i, feature in enumerate(features):
        counts = df[feature].value_counts().sort_index()
        percents = counts / total_n * 100

        sns.barplot(
            x=counts.index.astype(str),
            y=counts.values,
            color=PLOT_COLORS[0],
            ax=axes[i]
        )

        axes[i].set_title(f"Boxplot of {feature}", fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold")
        axes[i].set_ylabel("Count")
        axes[i].set_xlabel("")

        for j, (count, pct) in enumerate(zip(counts.values, percents.values)):
            axes[i].text(
                j,
                count,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()


def _plot_continues_features(df: DataFrame | Any, features: list[Any]):
    fig, axes = plt.subplots(len(features), 2, figsize=(12, 3 * len(features)))

    if len(features) == 1:
        axes = axes.reshape(1, 2)

    for i, col in enumerate(features):
        sns.histplot(
            df[col],
            kde=True,
            ax=axes[i, 0],
            color=PLOT_COLORS[0]
        )
        axes[i, 0].set_title(f"Distribution of {col}\n", fontsize = PLOT_TITLE_FONTSIZE, fontweight="bold")
        axes[i, 0].set_xlabel("")

        sns.boxplot(
            x=df[col],
            ax=axes[i, 1],
            color=PLOT_COLORS[0],
            width=0.5,
            fliersize=3
        )
        axes[i, 1].set_title(f"Boxplot of {col}\n", fontsize = PLOT_TITLE_FONTSIZE, fontweight="bold")
        axes[i, 1].set_xlabel("")

    plt.tight_layout()
    plt.show()


def _plot_correlation_matrix(df: DataFrame):
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    for col in numeric_cols:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    corr_matrix = df[numeric_cols].corr()

    rows_to_keep = corr_matrix.index[1:]
    cols_to_keep = corr_matrix.columns[:-1]
    corr_matrix_reduced = corr_matrix.loc[rows_to_keep, cols_to_keep]

    mask = np.triu(np.ones_like(corr_matrix_reduced, dtype=bool), 1)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        corr_matrix_reduced,
        mask=mask,
        annot=corr_matrix_reduced,
        cmap='Blues',
        fmt=".2f",
        square=True,
        cbar=False,
        xticklabels=corr_matrix_reduced.columns,
        yticklabels=corr_matrix_reduced.index,
    )
    plt.title("Feature Correlation Matrix\n", fontsize = TITLE_FONTSIZE, fontweight="bold")
    plt.show()


def _plot_target_correlation(df: pd.DataFrame, target: str, features: list[str]):
    """
    Plots a horizontal bar chart showing the correlation of all numeric features with the target feature.

    Parameters:
        df (pd.DataFrame): Input dataframe containing features and target.
        target (str): Name of the target column.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in features.")

    features.remove(target)

    correlations = df[features].corrwith(df[target])

    plt.figure(figsize=(8, len(features) * 0.5 + 2))
    plt.barh(correlations.index, correlations.values, color=PLOT_COLORS[0])
    plt.xlabel("Correlation with Target")
    plt.ylabel("Features")
    plt.title(f"Feature Correlation with '{target}'")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()