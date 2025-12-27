import pandas as pd
import numpy as np

def clean_data(df):
    """Basic cleaning: fix timestamps, remove negatives, fill missing."""
    df = df.copy()

    # Drop rows with missing required fields
    df = df.dropna(subset=["timestamp", "amount", "merchant", "category", "device"])

    # Remove negative or zero amounts
    df = df[df["amount"] > 0]

    # Ensure timestamp is datetime
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"])

    return df


def engineer_features(df):
    """Create all ML features."""
    df = df.copy()

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] <= 6) | (df["hour"] >= 22)).astype(int)

    # Amount transforms
    df["amount_log"] = np.log1p(df["amount"])
    df["high_amount_flag"] = (df["amount"] > 500).astype(int)

    # Aggregate user behaviour
    df = df.sort_values(by=["user_id", "timestamp"])

    df["user_txn_count_24h"] = df.groupby("user_id")["timestamp"].transform(
        lambda ts: ts.diff().dt.total_seconds().lt(24*3600).cumsum()
    )

    df["user_prev_amount"] = df.groupby("user_id")["amount"].shift(1).fillna(0)

    # Device/merchant/category one-hot encoding
    df = pd.get_dummies(df, columns=["merchant", "category", "device"], drop_first=True)

    return df


def prepare_dataset(df):
    """Return X (features) and y (label) ready for modeling."""
    df = df.copy()
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud", "timestamp", "transaction_id", "ip_address"])

    return X, y
