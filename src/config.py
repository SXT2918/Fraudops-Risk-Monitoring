"""Central configuration for FraudOps.

Keeping paths and constants here prevents hardcoded values from being
scattered across the project. This makes the repo easier to maintain and
upgrade later, for example if SQLite is replaced by PostgreSQL.
"""

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

DEFAULT_RAW_FILE = SAMPLE_DATA_DIR / "sample_transactions.csv"
CLEAN_TRANSACTIONS_FILE = PROCESSED_DATA_DIR / "clean_transactions.csv"
DATABASE_PATH = BASE_DIR / "fraudops.sqlite3"

MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Schema expected from the raw transaction source.
REQUIRED_COLUMNS = [
    "transaction_id",
    "user_id",
    "amount",
    "merchant_category",
    "transaction_time",
    "location",
    "is_fraud",
]

# Fraud risk threshold defaults. These are business rules, not model rules.
LOW_RISK_MAX = 0.30
HIGH_RISK_MIN = 0.70
DEFAULT_DECISION_THRESHOLD = HIGH_RISK_MIN
MODEL_EVALUATION_THRESHOLD = 0.50

# Reproducible model evaluation settings.
RANDOM_STATE = 42
TEST_SIZE = 0.30
THRESHOLD_COMPARISON_VALUES = [0.10, 0.20, 0.30, 0.50, 0.70]

# Simple assumption for business impact reporting.
# True-positive fraud caught => prevented loss equals transaction amount.
PREVENTED_LOSS_MULTIPLIER = 1.0
