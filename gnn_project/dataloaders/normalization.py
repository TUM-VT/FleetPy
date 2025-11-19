import os
import shutil
import pandas as pd
import numpy as np
from gnn_project.config import Config as cfg
import logging

logger = logging.getLogger(__name__)

def clean_normalization_directory(stats_dir: str) -> None:
    """
    Cleans the normalization statistics directory by removing existing files
    and recreating the directory if it doesn't exist.
    
    Args:
        stats_dir (str): Path to the directory containing normalization statistics
    """
    if os.path.exists(stats_dir):
        shutil.rmtree(stats_dir)
    os.makedirs(stats_dir, exist_ok=True)


def load_normalization_statistics(stats_dir: str) -> dict[str, dict]:
    """
    Loads saved normalization statistics from parquet files.

    Args:
        stats_dir (str): Path to the directory containing normalization statistics

    Returns:
        dict[str, dict]: Dictionary containing 'means', 'stds', 'mins', and 'maxs' dictionaries for each feature.
              The keys in these dictionaries include the feature type prefix (e.g., 'req_feature', 'veh_feature')
    """
    # Read parquet files and convert to dictionaries
    means_df = pd.read_parquet(os.path.join(stats_dir, "means.parquet"))
    stds_df = pd.read_parquet(os.path.join(stats_dir, "stds.parquet"))
    mins_df = pd.read_parquet(os.path.join(stats_dir, "mins.parquet"))
    maxs_df = pd.read_parquet(os.path.join(stats_dir, "maxs.parquet"))

    # Convert to dictionaries preserving the prefixed column names
    means = means_df.iloc[:, 0].to_dict()
    stds = stds_df.iloc[:, 0].to_dict()
    mins = mins_df.iloc[:, 0].to_dict()
    maxs = maxs_df.iloc[:, 0].to_dict()

    return {'means': means, 'stds': stds, 'mins': mins, 'maxs': maxs}


def get_feature_type(series: pd.Series, column_name: str = None) -> str:
    """
    Determine the type of feature for normalization purposes.

    Args:
        series (pd.Series): The column to check
        column_name (str, optional): Name of the column for explicit matching

    Returns:
        str: Feature type ('binary', 'categorical', 'continuous', or 'metadata')
    """
    import re

    # 1. Metadata columns (always exclude from normalization)
    metadata_patterns = [
        'id$', 'timestep', 'source', 'target', cfg.label_key
    ]
    if column_name and any(re.search(pattern, column_name.lower()) for pattern in metadata_patterns):
        return 'metadata'

    # 2. Binary indicators and flags
    binary_patterns = [
        'locked$', '^is_', 'feasibility', cfg.init_label_key
    ]
    if column_name and any(re.search(pattern, column_name.lower()) for pattern in binary_patterns):
        return 'binary'

    # 3. Categorical features (one-hot encoded or discrete classes)
    categorical_patterns = [
        'status_[0-9]+$', 'type_[a-z]+$',  # one-hot encoded columns
    ]
    if column_name and any(re.search(pattern, column_name.lower()) for pattern in categorical_patterns):
        return 'categorical'

    # 4. Check actual values for binary/categorical nature
    unique_vals = pd.unique(series.dropna())

    # If boolean type or only contains 0/1
    if series.dtype == bool or (len(unique_vals) <= 2 and set(unique_vals) <= {0, 1, 0.0, 1.0}):
        # Exception: Don't treat counts/ratios as binary even if they temporarily have only 0/1
        if column_name and not any(x in column_name.lower() for x in ['ratio', 'count', 'degree', 'centrality', 'time', 'total', 'common', 'exclusive', 'competition']):
            return 'binary'

    # If small number of unique integers, likely categorical
    # if series.dtype in ['int32', 'int64'] and len(unique_vals) < 10:
    #     return 'categorical'

    # 5. Continuous features (everything else)
    # This includes:
    # - Spatial features (lat, lon, distances)
    # - Temporal features (times, durations)
    # - Network metrics (degrees, centrality)
    # - Scores and ratios (efficiency, overlap)
    return 'continuous'


def normalize_features(df: pd.DataFrame,
                       means: dict,
                       stds: dict,
                       exclude_columns: list = None,
                       prefix: str = None) -> pd.DataFrame:
    """
    Applies z-score normalization (mean=0, std=1) to DataFrame features.

    Args:
        df (pd.DataFrame): Input DataFrame to normalize
        means (dict): Dictionary of means for each feature
        stds (dict): Dictionary of standard deviations for each feature
        exclude_columns (list, optional): Columns to exclude from normalization
        prefix (str, optional): Prefix used in the statistics dictionary (e.g., 'req_', 'veh_')

    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    if exclude_columns is None:
        exclude_columns = []

    # Create a copy to avoid modifying the original DataFrame
    df_norm = df.copy()

    # Get columns to normalize (numeric columns not in exclude list and not binary)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Combine explicit exclude list with binary columns
    cols_to_normalize = [
        col for col in numeric_cols if col not in exclude_columns]
    logger.debug(f"Columns to normalize: {cols_to_normalize} for {prefix}")

    # Apply z-score normalization
    for col in cols_to_normalize:
        if col in means and col in stds:
            std = stds[col]
            # Avoid division by zero
            if std > 0:
                df_norm[col] = (df[col] - means[col]) / std
            else:
                df_norm[col] = df[col] - means[col]

    return df_norm
