from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Update if your file name differs
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

TARGET_COL = "Churn"

RANDOM_SEED = 42
TEST_SIZE = 0.2

OUTPUT_MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = OUTPUT_REPORTS_DIR / "figures"
