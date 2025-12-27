"""
src/data_loader.py

Functions:
- generate_sample_transactions(path, n=1000, seed=42)
- load_transactions(path, required_cols=None)

Creates a synthetic transactions CSV and provides a robust loader that validates schema.
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "merchant",
    "category",
    "device",
    "ip_address",
    "is_fraud"
]

def generate_sample_transactions(path="data/transactions.csv", n=2000, seed=42):
    """
    Generate a synthetic transactions CSV with n rows and save to path.
    The generated file is realistic enough for development and testing.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    start_time = datetime.now() - timedelta(days=30)
    merchants = ["Walmart", "Amazon", "Flipkart", "BigMart", "LocalStore", "ElectroShop"]
    categories = ["grocery", "electronics", "fashion", "utilities", "entertainment"]
    devices = ["mobile", "desktop", "tablet"]
    ip_base = "192.168.{}.{}"

    rows = []
    for i in range(1, n + 1):
        txn_id = f"TXN{i:07d}"
        user_id = f"U{random.randint(1, 800):05d}"
        ts = start_time + timedelta(seconds=random.randint(0, 30 * 24 * 3600))
        amount = round(float(np.random.exponential(scale=80.0)) + random.choice([0, 5, 10, 20]), 2)
        merchant = random.choice(merchants)
        category = random.choice(categories)
        device = random.choice(devices)
        ip_address = ip_base.format(random.randint(0, 255), random.randint(0, 255))
        # simple heuristic for fraud: high amounts + night transactions + certain merchants
        hour = ts.hour
        risk_score = 0
        if amount > 1000:
            risk_score += 2
        if hour < 6 or hour > 22:
            risk_score += 1
        if merchant in ["LocalStore", "ElectroShop"] and amount > 500:
            risk_score += 1
        is_fraud = 1 if (random.random() < 0.05 * (1 + risk_score)) else 0

        rows.append({
            "transaction_id": txn_id,
            "user_id": user_id,
            "timestamp": ts.isoformat(sep=" "),
            "amount": amount,
            "merchant": merchant,
            "category": category,
            "device": device,
            "ip_address": ip_address,
            "is_fraud": is_fraud
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[data_loader] Wrote synthetic transactions to: {path}  (rows={len(df)})")
    return path

def load_transactions(path="data/transactions.csv", required_cols=None):
    """
    Load transactions CSV, validate required columns and basic dtypes, and return DataFrame.
    Raises ValueError if required columns are missing.
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # basic type conversions and sanity checks
    # timestamp -> datetime
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        # fallback: if timestamps are epoch ints
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        except Exception as e:
            raise ValueError("Failed to parse 'timestamp' column") from e

    # amount -> numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if df["amount"].isna().any():
        print("[data_loader] Warning: some 'amount' values could not be parsed and are NaN")

    # is_fraud -> integer (0/1)
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    df["is_fraud"] = df["is_fraud"].clip(0, 1)

    # basic sanity prints
    print(f"[data_loader] Loaded {len(df)} rows. Fraud cases: {int(df['is_fraud'].sum())}")

    return df

# quick internal test helper (callable)
def quick_check(path="data/transactions.csv", n=500):
    """
    Convenience helper: generate sample file and then load it.
    Returns the loaded DataFrame.
    """
    generate_sample_transactions(path=path, n=n)
    df = load_transactions(path=path)
    return df
