from typing import List, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

pd.set_option("display.max_columns", 500)


# Displays the total number of null values per column and the % of those within the column of the provided DataFrame
# In case a col name is parsed as a parameter, only that column will be printed
def null_percentage(df: pd.DataFrame, col: str = None):
    total_rows = len(df)
    df_cols = df.columns
    if col in df_cols:
        null_count = df[col].isnull().sum()
        percentage_null = (null_count / total_rows) * 100
        print(f"{col}: total -> {null_count}, percentage -> {percentage_null:.2f}%")
    elif (col not in df_cols) & (col is not None):
        print(f'Column "{col}" not in Dataframe')
    else:
        for col in df.columns:
            null_count = df[col].isnull().sum()
            percentage_null = (null_count / total_rows) * 100
            print(f"{col}: total -> {null_count}, percentage -> {percentage_null:.2f}%")


# One-Hot Encoder for specified categorical features
# Takes DataFrame and list of columns to encode as parameters
def one_hot_encoding(
    df: pd.DataFrame, cols: Union[List[str], pd.DataFrame]
) -> pd.DataFrame:
    if isinstance(cols, pd.DataFrame):
        # If 'cols' is a DataFrame, assume the column names are to be used
        cols = cols.columns.tolist()
    if all(col in df.columns for col in cols):
        return pd.DataFrame(pd.get_dummies(df, columns=cols, drop_first=False))
    else:
        print("Make sure all columns are on the DataFrame")
        return df


# Label Encoder for specified Categorical features
# Takes Dataframe and list of columns to encode as parameters
# def label_encoder(df:pd.DataFrame, cols: List[str]) -> pd.DataFrame:
def label_encoding(
    df: pd.DataFrame, cols: Union[List[str], pd.DataFrame], cols_dict: dict
) -> pd.DataFrame:
    if isinstance(cols, pd.DataFrame):
        # If 'cols' is a DataFrame, assume the column names are to be used
        cols = cols.columns.tolist()
    if all(col in df.columns for col in cols):
        for col in cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        return df
    else:
        print("Make sure all columns are on the DataFrame")
        return df


# Violin plot specified numeric columns in DataFrame
def violin_plot_numeric(df: pd.DataFrame, cols: Union[List[str], pd.DataFrame]) -> None:
    if isinstance(cols, pd.DataFrame):
        # If 'cols' is a DataFrame, assume the column names are to be used
        cols = cols.columns.tolist()
    if all(col in df.columns for col in cols):
        n_rows = len(cols)
        fig, axs = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
        sns.set_palette([(0.8, 0.56, 0.65), "crimson", (0.99, 0.8, 0.3)])

        for i, col in enumerate(cols):
            sns.violinplot(x="Status", y=col, data=df, ax=axs[i])
            axs[i].set_title(f"{col.title()} Distribution by Target", fontsize=14)
            axs[i].set_xlabel("Outcome", fontsize=12)
            axs[i].set_ylabel(col.title(), fontsize=12)
            sns.despine()

        fig.tight_layout()

        plt.show()


def value_counts_all_cols(
    df: pd.DataFrame, cols: Union[List[str], pd.DataFrame] = None
) -> None:
    if cols is not None:
        if isinstance(cols, pd.DataFrame):
            # If 'cols' is a DataFrame, assume the column names are to be used
            cols = cols.columns.tolist()
        for col in cols:
            print(f"Value Counts for : {df[col].value_counts()}\n")
    else:
        for col in df.columns:
            print(f"Value Counts for : {df[col].value_counts()}\n")


# Finds correlation between variables and creates a correlation heatmap.
# Returns a correlation num of all variables in DataFrame
# Takes a DataFrame as a parameter
def corr_heatmap(df: pd.DataFrame) -> float:
    df_numeric = df.select_dtypes(include=["int64", "float64", "boolean"])
    corr_num = df_numeric.corr()
    # Create heatmap
    plt.figure(figsize=(22, 10))
    sns.heatmap(corr_num, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    return corr_num


def train_models(
    models: list[dict], data: pd.DataFrame, label, n_splits=5, n_repeats=1, seed=43
):
    """Trains Models. The optimal parameters
    should be retrieved from previous runs e.g. GridSearchCV etc."""
    train_scores = {}
    pbar = tqdm(models)
    for model in pbar:

        model_str = model["name"]
        model_est = model["model"]
        model_feats = model["feats"]

        pbar.set_description(f"Processing {model_str}...")

        train_scores[model_str] = []

        skf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )

        for i, (train_idx, val_idx) in enumerate(
            skf.split(data[model_feats], data[label])
        ):
            pbar.set_postfix_str(f"Fold {i+1}/{n_splits}")
            # Resetting index to ensure valid indices
            train_idx = data[model_feats].index[train_idx]
            val_idx = data[model_feats].index[val_idx]
            X_train, y_train = (
                data[model_feats].loc[train_idx],
                data[label].loc[train_idx],
            )

            if model_str in ["lgb_cl"]:
                callbacks = [
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0),
                ]
                model_est.fit(X_train, y_train, callbacks=callbacks)
            elif model_str in ["xgb_cl", "cat_cl"]:
                model_est.fit(X_train, y_train, verbose=0)
            elif model_str in ["voting_clf"]:
                pass  # TODO: find a solution
            else:
                model_est.fit(X_train, y_train)

            train_preds = model_est.predict_proba(X_train[model_feats])
            train_score = log_loss(y_train, train_preds)
            train_scores[model_str].append(train_score)

    return models, pd.DataFrame(train_scores)


def validate_models(
    models: list[dict], data: pd.DataFrame, label, n_splits=5, n_repeats=1, seed=42
):
    """Run models and test them on validation sets. The optimal parameters
    should be retrieved from previous runs e.g. GridSearchCV etc."""

    # TODO: the model dicts should contain the FEATS (since different FEATS should be used)

    train_scores, val_scores = {}, {}

    pbar = tqdm(models)
    for model in pbar:

        # Model needs to be a dict (before tuple) since I need a mutable datatype
        # to insert the average validation score in the end
        model_str = model["name"]
        model_est = model["model"]
        model_feats = model["feats"]

        pbar.set_description(f"Processing {model_str}...")

        train_scores[model_str] = []
        val_scores[model_str] = []

        # I think I should drop the seed when I blend the models together
        # -> they will be trained on different datasets
        skf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )

        for i, (train_idx, val_idx) in enumerate(
            skf.split(data[model_feats], data[label])
        ):
            pbar.set_postfix_str(f"Fold {i + 1}/{n_splits}")
            # Resetting index to ensure valid indices
            train_idx = data[model_feats].index[train_idx]
            val_idx = data[model_feats].index[val_idx]
            X_train, y_train = (
                data[model_feats].loc[train_idx],
                data[label].loc[train_idx],
            )
            # X_train, y_train = data.loc[train_idx, model_feats], data.loc[train_idx, label]
            X_val, y_val = data[model_feats].loc[val_idx], data[label].loc[val_idx]

            # print(X_train.dtypes)
            if model_str in ["lgb_cl"]:
                callbacks = [
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0),
                ]
                model_est.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks
                )
            elif model_str in ["xgb_cl", "cat_cl"]:
                model_est.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            elif model_str in ["voting_clf"]:
                pass  # TODO: find a solution
            else:
                model_est.fit(X_train, y_train)

            train_preds = model_est.predict_proba(X_train[model_feats])
            valid_preds = model_est.predict_proba(X_val[model_feats])
            train_score = log_loss(y_train, train_preds)
            val_score = log_loss(y_val, valid_preds)

            train_scores[model_str].append(train_score)
            val_scores[model_str].append(val_score)

            # print(f"{model_str} | Fold {i + 1} | " +
            #      f"Train log_loss: {round(train_score, 4)} | " +
            #      f"Valid log_loss: {round(val_score, 4)}")

        model["avg_val_score"] = np.mean(val_scores[model_str])

    return models, pd.DataFrame(train_scores), pd.DataFrame(val_scores)
