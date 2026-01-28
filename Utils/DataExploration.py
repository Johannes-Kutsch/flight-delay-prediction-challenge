import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Any

from pandas import DataFrame
from sklearn.feature_selection import mutual_info_regression

from Utils.HelperFunctions import _aggregate_df
from Utils.Styling import PLOT_COLORS, PLOT_TITLE_FONTSIZE, TITLE_FONTSIZE


def explore_dataframe(df: pd.DataFrame, target_feature: list[Any] = None, target_feature_name:str = None, feature_mask:list[str] = None,
                      plot_data_spread = True, plot_corr_matrix : bool = True, plot_pairs :bool = True, plot_target_correlation : bool = True, categorical_feature_names:list[str] = None,):
    sns.set_style("whitegrid")
    df, has_target_feature, target_feature_name = _aggregate_df(df, feature_mask, target_feature, target_feature_name)

    categorical_features, continues_features = _sort_features(df, categorical_feature_names)

    has_continues_features = len(continues_features) > 0
    has_categorical_features = len(categorical_features) > 0

    if plot_data_spread and has_continues_features:
        _plot_continues_features(df, continues_features, has_target_feature, target_feature_name)

    if plot_data_spread and has_categorical_features:
        _plot_categorical_features(df, categorical_features, has_target_feature, target_feature_name)

    if plot_pairs:
        if has_target_feature:
            _plot_pairs_with_target(df, categorical_features, continues_features, target_feature_name)
        else:
            _plot_pairs(df, categorical_features, continues_features)

    if plot_corr_matrix and len(continues_features) > 1:
        _plot_correlation_matrix(df, continues_features)

    if plot_target_correlation and has_target_feature:
        _plot_target_correlation(df, target_feature_name, continues_features, categorical_features)


def _sort_features(df, categorical_feature_names) -> tuple[
    list[Any], list[Any]]:
    categorical_features = []
    continues_features = []

    for col in df.columns:
        if categorical_feature_names is not None and col in categorical_feature_names:
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
            categorical_features.append(col)

        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
            categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            continues_features.append(col)
        elif pd.api.types.is_categorical_dtype(df[col]):
            categorical_features.append(col)
        elif pd.api.types.is_string_dtype(df[col]):
            categorical_features.append(col)
    return categorical_features, continues_features

def _plot_pairs(df: pd.DataFrame, categorical_features: list, continues_features: list):
    analyze_cols = categorical_features + continues_features

    df = df[analyze_cols]
    df = df.dropna()

    plot = sns.pairplot(
        df,
        corner=True,
        diag_kind="kde",
        plot_kws={"alpha": 0.6},
        palette=PLOT_COLORS,
        height=3.2,
        aspect=1.1
    )

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

def _plot_pairs_with_target(df: pd.DataFrame, categorical_features: list, continues_features: list, target_feature_name: str):
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

def _plot_categorical_features(df: DataFrame, features: list[Any], has_target_feature, target_feature_name: str):
    is_target_numeric = has_target_feature and pd.api.types.is_numeric_dtype(df[target_feature_name])

    n_cols = 3 if has_target_feature else 1

    fig, axes = plt.subplots(len(features), n_cols, figsize=(6 * n_cols, 3 * len(features)))

    if len(features) == 1:
        axes = axes.reshape(1, -1)

    total_n = len(df)

    for i, feature in enumerate(features):
        counts = df[feature].value_counts().sort_index()
        percents = counts / total_n * 100

        sns.barplot(x=counts.index.astype(str), y=counts.values, color=PLOT_COLORS[0], ax=axes[i, 0])

        axes[i, 0].set_title(f"Distribution of {feature}\n", fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold")
        axes[i, 0].set_ylabel("Count")
        axes[i, 0].set_xlabel("")

        for j, (count, pct) in enumerate(zip(counts.values, percents.values)):
            axes[i, 0].text(j, count, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

        if has_target_feature:
            sns.stripplot(data=df, x=feature, y=target_feature_name, jitter=True, alpha=0.6, color=PLOT_COLORS[0],
                          ax=axes[i, 1])

            axes[i, 1].set_title(f"{target_feature_name} per {feature}\n", fontsize=PLOT_TITLE_FONTSIZE,
                                 fontweight="bold")
            axes[i, 1].set_xlabel("")

        if is_target_numeric:
            sns.boxplot(data=df, x=feature, y=target_feature_name, color=PLOT_COLORS[0], ax=axes[i, 2],
                        showfliers=False)

            stats = df.groupby(feature)[target_feature_name].agg(["mean", "std"]).reset_index()

            axes[i, 2].errorbar(x=np.arange(len(stats)), y=stats["mean"], yerr=stats["std"], fmt="o", color="black",
                                capsize=4, label="Mean ± STD")
            axes[i, 2].set_title(f"{target_feature_name} per {feature}\n", fontsize=PLOT_TITLE_FONTSIZE,
                                 fontweight="bold")
            axes[i, 2].set_xlabel("")

    plt.tight_layout()
    plt.show()

def _plot_continues_features(df: DataFrame | Any, features: list[Any], has_target_feature, target_feature_name: str):
    n_cols = 3 if has_target_feature else 2
    fig, axes = plt.subplots(len(features), n_cols, figsize=(18, 3 * len(features)))
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

        sns.boxplot(
            x=df[col],
            ax=axes[i, 1],
            color=PLOT_COLORS[0],
            width=0.5,
            fliersize=3
        )
        axes[i, 1].set_title(f"Boxplot of {col}\n", fontsize = PLOT_TITLE_FONTSIZE, fontweight="bold")

        if has_target_feature:
            # if col != target_feature_name:
            sns.scatterplot(
                x=df[col],
                y=df[target_feature_name],
                ax=axes[i, 2],
                color=PLOT_COLORS[0],
                alpha=0.6,
            )
            axes[i, 2].set_title(f"Scatterplot of {col} x {target_feature_name}\n", fontsize=PLOT_TITLE_FONTSIZE,
                                 fontweight="bold")
            # else:
            # fig.delaxes(axes[i, 2])



    plt.tight_layout()
    plt.show()


def _plot_correlation_matrix(df: DataFrame, continuous_features: list[str]):
    corr_matrix = df[continuous_features].corr()

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
    plt.title("Continuous Feature Correlation Matrix\n", fontsize = TITLE_FONTSIZE, fontweight="bold")
    plt.show()


def _plot_target_correlation(df: pd.DataFrame, target: str, continuous_features: list[str], categorical_features: list[str]):
    """
    Plots a horizontal bar chart showing correlation of numeric features and categorical features (as dummies) with the target.

    Parameters:
        df (pd.DataFrame): Input dataframe containing features and target.
        target (str): Name of the target column.
        continuous_features (list[str]): List of continuous/numeric feature column names.
        categorical_features (list[str]): List of categorical feature column names (object/str or bool).
    """
    continuous_features = [f for f in continuous_features if f != target]
    categorical_features = [f for f in categorical_features if f != target]

    scores = {}

    if continuous_features:
        cont_corr = df[continuous_features].corrwith(df[target])
        scores.update(cont_corr.to_dict())

    for cat_feat in categorical_features:
        X = df[[cat_feat]].copy()
        if not (
            pd.api.types.is_categorical_dtype(X[cat_feat])
            or pd.api.types.is_bool_dtype(X[cat_feat])
            or pd.api.types.is_integer_dtype(X[cat_feat])
        ):
            X[cat_feat] = X[cat_feat].astype("category")

        # Label-Encoding (MI braucht numerisch)
        X_encoded = X[cat_feat].cat.codes.to_frame()

        mi = mutual_info_regression(
            X_encoded,
            df[target],
            discrete_features=True,
            random_state=42,
        )[0]

        scores[cat_feat] = mi

    score_series = pd.Series(scores).sort_values()
    plt.figure(figsize=(8, len(scores) * 0.5 + 2))
    plt.barh(score_series.index, score_series.values, color='skyblue')
    plt.xlabel("Correlation with Target")
    plt.ylabel("Features")
    plt.title(f"Feature Correlation with '{target}'\n", fontsize = TITLE_FONTSIZE, fontweight="bold")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()