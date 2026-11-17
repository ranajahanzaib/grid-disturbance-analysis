"""Grouped and stratified dataset splitting with leakage prevention."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    target_column: str = "disturbance_type",
    group_column: str | None = "event_id",
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Split dataset into Train, Validation, and Test partitions without data leakage.

    Returns:
        (train_df, val_df, test_df, strategy_used)
    """
    n = len(df)
    if n < 5:
        return df, df, df, "insufficient_data"

    # Check if group_column exists and has sufficient distinct groups
    use_grouping = False
    if group_column and group_column in df.columns:
        num_groups = df[group_column].nunique()
        if num_groups >= 6:
            use_grouping = True

    if use_grouping:
        strategy = f"event_grouped_by_{group_column}"
        logger.info(f"Using leakage-safe grouped split by {group_column}")

        # 1. Split out test set by group
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df[group_column]))

        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # 2. Split train and validation by group
        val_relative = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=random_seed)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[group_column]))

        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()

    else:
        strategy = "stratified_by_target"
        logger.warning("Event grouping not available. Falling back to stratified splitting.")

        # Check if each class has at least 2 samples
        class_counts = df[target_column].value_counts()
        if (class_counts < 2).any():
            # Random split fallback
            idx = np.random.default_rng(random_seed).permutation(n)
            n_test = int(n * test_size)
            n_val = int(n * val_size)
            test_df = df.iloc[idx[:n_test]].copy()
            val_df = df.iloc[idx[n_test : n_test + n_val]].copy()
            train_df = df.iloc[idx[n_test + n_val :]].copy()
            return train_df, val_df, test_df, "random_fallback"

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_val_idx, test_idx = next(sss_test.split(df, df[target_column]))

        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()

        val_relative = val_size / (1.0 - test_size)
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_relative, random_state=random_seed
        )
        train_idx, val_idx = next(sss_val.split(train_val_df, train_val_df[target_column]))

        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()

    # Assign partition tags
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return train_df, val_df, test_df, strategy
