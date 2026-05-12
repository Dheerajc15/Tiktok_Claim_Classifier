
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def load_data(path: str) -> pd.DataFrame:
    """Load the raw TikTok dataset from CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Drop rows with missing claim_status (target)
    - Drop rows with missing transcription text
    - Reset index
    """
    df = df.dropna(subset=["claim_status"]).copy()
    df = df.dropna(subset=["video_transcription_text"]).copy()
    df = df.reset_index(drop=True)
    return df


def encode_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features:
    - claim_status: claim=1, opinion=0
    - verified_status: verified=1, not verified=0
    - author_ban_status: one-hot encoded
    """
    df = df.copy()
    df["claim_status"] = df["claim_status"].map({"claim": 1, "opinion": 0})
    df["verified_status"] = df["verified_status"].map(
        {"verified": 1, "not verified": 0}
    )
    df = pd.get_dummies(df, columns=["author_ban_status"], drop_first=True)
    return df


def split_features_target(df: pd.DataFrame):
    """Separate features (X) and target (y). Drops ID columns."""
    drop_cols = ["#", "video_id", "claim_status"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["claim_status"]
    return X, y


def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """
    Stratified 60/20/20 train/val/test split.
    First splits off 20% for test, then 25% of remaining for val (= 20% of total).
    """
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def vectorize_text(
    X_train, X_val, X_test, text_col="video_transcription_text", max_features=15
):
    """
    Fit CountVectorizer (bigrams + trigrams) on train only, transform all splits.
    Returns transformed feature matrices and the fitted vectorizer.
    """
    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        max_features=max_features,
        stop_words="english",
    )

    train_text = vectorizer.fit_transform(X_train[text_col]).toarray()
    val_text = vectorizer.transform(X_val[text_col]).toarray()
    test_text = vectorizer.transform(X_test[text_col]).toarray()

    feature_names = vectorizer.get_feature_names_out()
    train_text_df = pd.DataFrame(train_text, columns=feature_names, index=X_train.index)
    val_text_df = pd.DataFrame(val_text, columns=feature_names, index=X_val.index)
    test_text_df = pd.DataFrame(test_text, columns=feature_names, index=X_test.index)

    X_train_final = pd.concat([X_train.drop(columns=[text_col]), train_text_df], axis=1)
    X_val_final = pd.concat([X_val.drop(columns=[text_col]), val_text_df], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=[text_col]), test_text_df], axis=1)

    return X_train_final, X_val_final, X_test_final, vectorizer


def prepare_data(data_path: str):
    """
    End-to-end preprocessing: load -> clean -> encode -> split -> vectorize.
    Returns all splits ready for modeling, plus the fitted vectorizer.
    """
    df = load_data(data_path)
    df = clean_data(df)
    df = encode_targets(df)
    X, y = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train, X_val, X_test, vectorizer = vectorize_text(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer