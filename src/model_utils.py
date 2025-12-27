import os
import joblib
import pandas as pd
from src.preprocess import clean_data, engineer_features, prepare_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    return joblib.load(path)

def _prepare_single_df(record: dict, model):
    df = pd.DataFrame([record])

    # timestamp handling
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # placeholder label
    df['is_fraud'] = 0

    # same preprocessing as training
    df = clean_data(df)
    df = engineer_features(df)
    X, _ = prepare_dataset(df)

    # align columns to what model saw during training
    expected_cols = model.named_steps['pre'].feature_names_in_
    X = X.reindex(columns=expected_cols, fill_value=0)

    return X

def predict_single(record: dict, model=None):
    if model is None:
        model = load_model()

    X = _prepare_single_df(record, model)
    pred = model.predict(X)[0]

    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)[0, 1]
    else:
        prob = None

    return int(pred), float(prob) if prob is not None else None

if __name__ == '__main__':
    m = load_model()
    sample = {
        'transaction_id': 'TXN_TEST',
        'user_id': 'U00001',
        'timestamp': '2025-11-01 12:00:00',
        'amount': 1500,
        'merchant': 'ElectroShop',
        'category': 'electronics',
        'device': 'mobile',
        'ip_address': '192.168.1.5'
    }
    print(predict_single(sample, m))
