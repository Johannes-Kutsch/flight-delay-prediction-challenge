from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import get_scorer_names
from sklearn.model_selection import cross_validate

from Utils.StringHelperFunctions import strip_ansi_codes
from Utils.Styling import RED, YEL, GRE, ORA, RES


def cross_validation_scoring(pipelines: object, X: object, y: object, folds: int = 5, scores: list[str] = None, verbose: bool = True,
                             order_by: str = None) -> dict [str, Any]:
    """
    Perform cross-validation for multiple pipelines and return a DataFrame with evaluation metrics.

    Parameters
    ----------
    pipelines : list of tuples
        List of (name, pipeline) tuples to evaluate.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Target variable.
    folds : int, default=5
        Number of cross-validation folds.
    order_by : str, default=None
        Order by Metric
    scores : dict or None, default=None
        Dictionary of scoring metrics If None, defaults to using using all score types
        e.g. {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
    verbose : Logs results to stdout.

    Returns
    -------
    dict
        Dictionary with cross-validated scores for each pipeline and metric.
    """

    if scores is None:
        scores = get_scores()

    results_dict = {}

    for pipe_name, pipe in pipelines:
        cv_results = cross_validate(pipe, X, y, cv=folds, scoring=scores, n_jobs=-1)

        for score_name in scores:
            results_dict.setdefault(pipe_name, {})
            results_dict[pipe_name][score_name] = cv_results[f"test_{score_name}"]

    if verbose:
        print_cross_validation_score_table(results_dict, _get_score_name(order_by))

    return results_dict

def gridsearch_to_cv_table_dict(grid_search, classifier_key="classifier"):
    """
    Convert GridSearchCV results to the format expected by print_cross_validation_score_table:
    results[model_name][metric] -> List of fold values
    parameters[model_name] -> List of Parameters
    Adds 'Parameters' as an extra metric.
    """
    cv_results = grid_search.cv_results_
    results = {}
    parameters = {}
    n_splits = grid_search.cv

    if isinstance(grid_search.scoring, dict):
        metrics = list(grid_search.scoring.keys())
        key_template = "split{fold}_test_{metric}"
        display_name = lambda m: m
    else:
        metrics = ["score"]
        key_template = "split{fold}_test_score"
        display_name = lambda _: grid_search.scoring

    for i, params in enumerate(cv_results["params"]):
        clf = params[classifier_key]
        model_name = f"{clf.__class__.__name__} #{i}"
        results[model_name] = {}

        for metric in metrics:
            split_scores = [
                cv_results[key_template.format(fold=j, metric=metric)][i]
                for j in range(n_splits)
            ]
            results[model_name][display_name(metric)] = split_scores

        parameters[model_name] = params

    return results, parameters

def print_cross_validation_score_table(cv_results_dict: dict, order_by: str = None, parameters: dict = None, precision: int = 2):
    """
    Print CV results as colored table: mean ± std per model × metric.
    cv_results_dict: [model][metric] -> List[values]
    """

    models = list(cv_results_dict)
    metrics = list(next(iter(cv_results_dict.values())))

    for model in models:
        model_results = cv_results_dict[model]
        for metric in metrics:
            model_results[metric] = np.abs(np.array(model_results[metric]))

    if order_by is not None:
        order_by = _get_score_name(order_by)
        ascending = _is_score_ascending(order_by)
        models = sorted(models, key=lambda m: np.mean(cv_results_dict[m][order_by]), reverse=ascending)

    color_lookup = _create_color_lookup(cv_results_dict)

    model_column_name = "Model"
    model_column_width = max(len(m) for m in models + [model_column_name]) + 2
    float_fmt = f"{{:.{precision}f}}"

    cell_widths = {}
    for metric in metrics:
        max_width = max(
            len(f"{float_fmt.format(np.mean(cv_results_dict[model][metric]))} (±{float_fmt.format(np.std(cv_results_dict[model][metric]))})")
            for model in models
        )
        cell_widths[metric] = max(max_width, len(_get_score_abbreviation(metric)))

    header = model_column_name.ljust(model_column_width)
    for metric in metrics:
        header += _get_score_abbreviation(metric).ljust(cell_widths[metric] + 2)
    print(header)

    for model in models:
        row = model.ljust(model_column_width)
        for metric in metrics:
            values = cv_results_dict[model][metric]
            mean = np.mean(values)
            std = np.std(values)

            mean_color, std_color = color_lookup[model][metric]
            cell = f"{mean_color}{float_fmt.format(mean)}{RES} ({std_color}±{float_fmt.format(std)}{RES})"

            padding = cell_widths[metric] - len(strip_ansi_codes(cell))
            row += cell + " " * (padding + 2)

        if parameters:
            row += ", ".join(f"{k}={v}" for k, v in parameters[model].items() if "__" in k or k == "classifier")

        print(row)

    print()

def get_scores(category: str = "all") -> list[str]:
    """
    Return all sklearn scores for one category.

    Parameters
    ----------
    category : str
        'classification', 'regression', 'clustering', or 'all'

    Returns
    -------
    list[str]
        [score_names]
    """

    category = category.lower()
    valid_categories = {"classification", "regression", "clustering", "all"}

    if category not in valid_categories:
        raise ValueError(f"category must be one of {valid_categories}")

    if category == "all":
        return sorted(get_scorer_names())

    scores = []

    for name in sorted(get_scorer_names()):
        scorer_category = _classify_score(name)
        if scorer_category != category:
            continue

        scores.append(name)

    return scores

def get_scores_by_name(names: list[str]) -> list[str]:
    return [_get_score_name(name.lower()) for name in names]

def print_score_descriptions(scores: list[str]):
    """
    Pretty-print sklearn score information.
    """

    print("=" * 90)

    for score in scores:
        is_ascending = "maximize" if _is_score_ascending(score) else "minimize (negated)"
        description = _SCORE_TYPE_DESCRIPTIONS.get(score, "No description available.")


        print(score)
        print(f"  Optimize     : {is_ascending}")
        print(f"  Description  : {description}")
        print(f"  Category     : {_classify_score(score)}")
        print("-" * 90)

def _is_score_ascending(name: str) -> bool:
    """
    Returns True if a lower value is better (ascending),
    False if a higher value is better (descending).
    """

    return not name.startswith("neg_") and not name in ["max_error"]

def _get_score_abbreviation(metric_name: str):
    if metric_name not in _COMMON_METRIC_ABBREVIATIONS:
        print(f"WARNING: {YEL}{metric_name} has no valid abbreviation.{RES}")

    return _COMMON_METRIC_ABBREVIATIONS.get(metric_name, metric_name)

def _get_score_name(metric_abbreviation: str):
    if metric_abbreviation not in _COMMON_METRIC_NAMES:
        print(f"{YEL}WARNING: {metric_abbreviation} has no valid name.{RES}")
    return _COMMON_METRIC_NAMES.get(metric_abbreviation, metric_abbreviation)


def _create_color_lookup(cv_results_dict: dict) -> dict:
    """
    Creates a nested dict for color coding:
    color_lookup[model][metric] = (mean_color, std_color)

    cv_results_dict: [model][metric] -> List[values]
    """
    color_lookup = {}
    metrics = list(next(iter(cv_results_dict.values())).keys())

    for model in cv_results_dict:
        color_lookup[model] = {}
        for metric in metrics:
            values_all_models = [np.mean(cv_results_dict[m][metric]) for m in cv_results_dict]
            std_all_models    = [np.std(cv_results_dict[m][metric])  for m in cv_results_dict]

            mean = np.mean(cv_results_dict[model][metric])
            std  = np.std(cv_results_dict[model][metric])

            ascending = _is_score_ascending(metric)

            mean_color = _get_color_by_rank(mean, values_all_models, ascending=ascending)
            std_color  = _get_color_by_rank(std, std_all_models, ascending=False)

            color_lookup[model][metric] = (mean_color, std_color)

    return color_lookup

def _get_color_by_rank(value, values, ascending=True):
    """
    Get Color by Rank in values:
    Best -> Green
    Second -> Yellow
    Middle -> no Color
    Second to Last -> Orange
    Last -> Red
    """
    n = len(values)
    sorted_vals = sorted(values, reverse=ascending)

    try:
        rank = sorted_vals.index(value)
    except ValueError:
        return ""  # fallback, falls value nicht gefunden

    if rank == 0:
        return GRE
    elif n > 3 and rank == 1:
        return YEL
    elif n > 4 and rank == n - 2:
        return ORA
    elif rank == n - 1:
        return RED
    else:
        return ""

def _classify_score(name: str) -> str:
    name_lower = name.lower()

    if name_lower in CLASSIFICATION_SCORES:
        return "classification"

    if name_lower in REGRESSION_SCORES:
        return "regression"

    if name_lower in CLUSTERING_SCORES:
        return "clustering"

    raise KeyError(
        f"{name} is no valid score type; "
        "use get_scores_by_category() to see valid types"
    )

CLASSIFICATION_SCORES = {
    "accuracy",
    "balanced_accuracy",
    "top_k_accuracy",
    "precision",
    "precision_macro",
    "precision_micro",
    "precision_samples",
    "precision_weighted",
    "recall",
    "recall_macro",
    "recall_micro",
    "recall_samples",
    "recall_weighted",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_samples",
    "f1_weighted",
    "roc_auc",
    "roc_auc_ovo",
    "roc_auc_ovo_weighted",
    "roc_auc_ovr",
    "roc_auc_ovr_weighted",
    "average_precision",
    "jaccard",
    "jaccard_macro",
    "jaccard_micro",
    "jaccard_samples",
    "jaccard_weighted",
    "matthews_corrcoef",
    "positive_likelihood_ratio",
    "neg_negative_likelihood_ratio",
    "neg_log_loss",
    "neg_brier_score",
}

REGRESSION_SCORES = {
    "r2",
    "explained_variance",
    "max_error",
    "neg_mean_absolute_error",
    "neg_mean_absolute_percentage_error",
    "neg_mean_squared_error",
    "neg_mean_squared_log_error",
    "neg_root_mean_squared_error",
    "neg_median_absolute_error",
    "neg_mean_gamma_deviance",
    "neg_mean_poisson_deviance",
}

CLUSTERING_SCORES = {
    "adjusted_rand_score",
    "rand_score",
    "adjusted_mutual_info_score",
    "mutual_info_score",
    "normalized_mutual_info_score",
    "homogeneity_score",
    "completeness_score",
    "v_measure_score",
    "fowlkes_mallows_score",
}

SCORE_TYPE_OVERRIDES = {
    "roc_auc": "classification",
    "average_precision": "classification",
    "neg_log_loss": "classification",
    "adjusted_rand_score": "clustering",
    "normalized_mutual_info_score": "clustering",
    "v_measure_score": "clustering",
    "fowlkes_mallows_score": "clustering",
}

_SCORE_TYPE_DESCRIPTIONS = {
    "accuracy": "Ratio of correctly classified samples.",
    "balanced_accuracy": "Accuracy adjusted for class imbalance.",
    "precision": "Fraction of predicted positives that are correct.",
    "recall": "Fraction of actual positives that are detected.",
    "f1": "Harmonic mean of precision and recall.",
    "roc_auc": "Area under the ROC curve.",
    "average_precision": "Area under the precision-recall curve.",
    "neg_log_loss": "Logarithmic loss on predicted probabilities.",
    "jaccard": "Intersection over union of label sets.",

    "r2": "Coefficient of determination.",
    "neg_mean_squared_error": "Mean squared error (penalizes large errors).",
    "neg_root_mean_squared_error": "Root mean squared error.",
    "neg_mean_absolute_error": "Mean absolute error.",
    "neg_median_absolute_error": "Median absolute error (outlier-robust).",
    "neg_mean_absolute_percentage_error": "Mean absolute percentage error.",
    "explained_variance": "Explained variance score.",

    "adjusted_rand_score": "Similarity between two cluster assignments.",
    "normalized_mutual_info_score": "Shared information between clusterings.",
    "v_measure_score": "Harmonic mean of homogeneity and completeness.",
    "fowlkes_mallows_score": "Precision/recall tradeoff for clustering.",
}

_COMMON_METRIC_ABBREVIATIONS = {
    "accuracy": "acc",
    "balanced_accuracy": "bacc",

    "precision": "prec",
    "precision_macro": "prec_macro",
    "precision_micro": "prec_micro",
    "precision_samples": "prec_s",
    "precision_weighted": "prec_w",

    "recall": "rec",
    "recall_macro": "rec_macro",
    "recall_micro": "rec_micro",
    "recall_samples": "rec_s",
    "recall_weighted": "rec_w",

    "f1": "f1",
    "f1_macro": "f1_macro",
    "f1_micro": "f1_micro",
    "f1_samples": "f1_s",
    "f1_weighted": "f1_w",

    "roc_auc": "auc",
    "roc_auc_ovo": "auc_ovo",
    "roc_auc_ovo_weighted": "auc_ovo_w",
    "roc_auc_ovr": "auc_ovr",
    "roc_auc_ovr_weighted": "auc_ovr_w",

    "average_precision": "ap",
    "top_k_accuracy": "topk_acc",

    "jaccard": "jac",
    "jaccard_macro": "jac_macro",
    "jaccard_micro": "jac_micro",
    "jaccard_samples": "jac_s",
    "jaccard_weighted": "jac_w",

    "r2": "r2",
    "explained_variance": "ev",

    "max_error": "maxe",
    "neg_mean_absolute_error": "mae",
    "neg_mean_absolute_percentage_error": "mape",
    "neg_mean_squared_error": "mse",
    "neg_root_mean_squared_error": "rmse",
    "neg_mean_squared_log_error": "msle",
    "neg_median_absolute_error": "medae",

    "neg_log_loss": "logloss",
    "neg_brier_score": "brier",

    "neg_mean_gamma_deviance": "mgd",
    "neg_mean_poisson_deviance": "mpd",

    "matthews_corrcoef": "mcc",

    "adjusted_rand_score": "ari",
    "rand_score": "ri",

    "adjusted_mutual_info_score": "ami",
    "mutual_info_score": "mi",
    "normalized_mutual_info_score": "nmi",

    "homogeneity_score": "homo",
    "completeness_score": "compl",
    "v_measure_score": "vmeasure",

    "fowlkes_mallows_score": "fmi",

    "positive_likelihood_ratio": "lr+",
    "neg_negative_likelihood_ratio": "lr-",
}

_COMMON_METRIC_NAMES = {
    # Accuracy
    "acc": "accuracy",
    "accuracy": "accuracy",
    "bacc": "balanced_accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "topk_acc": "top_k_accuracy",
    "top_k_accuracy": "top_k_accuracy",

    # Precision
    "prec": "precision",
    "precision": "precision",
    "prec_macro": "precision_macro",
    "precision_macro": "precision_macro",
    "prec_micro": "precision_micro",
    "precision_micro": "precision_micro",
    "prec_s": "precision_samples",
    "precision_samples": "precision_samples",
    "prec_w": "precision_weighted",
    "precision_weighted": "precision_weighted",

    # Recall
    "rec": "recall",
    "recall": "recall",
    "rec_macro": "recall_macro",
    "recall_macro": "recall_macro",
    "rec_micro": "recall_micro",
    "recall_micro": "recall_micro",
    "rec_s": "recall_samples",
    "recall_samples": "recall_samples",
    "rec_w": "recall_weighted",
    "recall_weighted": "recall_weighted",

    # F1
    "f1": "f1",
    "f1_macro": "f1_macro",
    "f1_micro": "f1_micro",
    "f1_s": "f1_samples",
    "f1_samples": "f1_samples",
    "f1_w": "f1_weighted",
    "f1_weighted": "f1_weighted",

    # ROC / AUC
    "auc": "roc_auc",
    "auroc": "roc_auc",
    "roc_auc": "roc_auc",
    "auc_ovo": "roc_auc_ovo",
    "roc_auc_ovo": "roc_auc_ovo",
    "auc_ovo_w": "roc_auc_ovo_weighted",
    "roc_auc_ovo_weighted": "roc_auc_ovo_weighted",
    "auc_ovr": "roc_auc_ovr",
    "roc_auc_ovr": "roc_auc_ovr",
    "auc_ovr_w": "roc_auc_ovr_weighted",
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",

    # Ranking / Probability
    "ap": "average_precision",
    "average_precision": "average_precision",
    "logloss": "neg_log_loss",
    "log_loss": "neg_log_loss",
    "neg_log_loss": "neg_log_loss",
    "brier": "neg_brier_score",
    "neg_brier_score": "neg_brier_score",

    # Jaccard
    "jac": "jaccard",
    "jaccard": "jaccard",
    "jac_macro": "jaccard_macro",
    "jaccard_macro": "jaccard_macro",
    "jac_micro": "jaccard_micro",
    "jaccard_micro": "jaccard_micro",
    "jac_s": "jaccard_samples",
    "jaccard_samples": "jaccard_samples",
    "jac_w": "jaccard_weighted",
    "jaccard_weighted": "jaccard_weighted",

    # Regression
    "r2": "r2",
    "ev": "explained_variance",
    "explained_variance": "explained_variance",
    "maxe": "max_error",
    "max_error": "max_error",
    "mae": "neg_mean_absolute_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
    "neg_mean_absolute_percentage_error": "neg_mean_absolute_percentage_error",
    "mse": "neg_mean_squared_error",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "msle": "neg_mean_squared_log_error",
    "rmsle": "neg_mean_squared_log_error",
    "neg_mean_squared_log_error": "neg_mean_squared_log_error",
    "medae": "neg_median_absolute_error",
    "neg_median_absolute_error": "neg_median_absolute_error",
    "mgd": "neg_mean_gamma_deviance",
    "neg_mean_gamma_deviance": "neg_mean_gamma_deviance",
    "mpd": "neg_mean_poisson_deviance",
    "neg_mean_poisson_deviance": "neg_mean_poisson_deviance",

    # Correlation / Association
    "mcc": "matthews_corrcoef",
    "matthews_corrcoef": "matthews_corrcoef",

    # Clustering
    "ari": "adjusted_rand_score",
    "adjusted_rand_score": "adjusted_rand_score",
    "ri": "rand_score",
    "rand_score": "rand_score",
    "ami": "adjusted_mutual_info_score",
    "adjusted_mutual_info_score": "adjusted_mutual_info_score",
    "mi": "mutual_info_score",
    "mutual_info_score": "mutual_info_score",
    "nmi": "normalized_mutual_info_score",
    "normalized_mutual_info_score": "normalized_mutual_info_score",
    "homo": "homogeneity_score",
    "homogeneity_score": "homogeneity_score",
    "compl": "completeness_score",
    "completeness_score": "completeness_score",
    "vmeasure": "v_measure_score",
    "v_measure_score": "v_measure_score",
    "fmi": "fowlkes_mallows_score",
    "fowlkes_mallows_score": "fowlkes_mallows_score",

    # Likelihood ratios
    "lr+": "positive_likelihood_ratio",
    "positive_likelihood_ratio": "positive_likelihood_ratio",
    "lr-": "neg_negative_likelihood_ratio",
    "neg_negative_likelihood_ratio": "neg_negative_likelihood_ratio",
}