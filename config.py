from pathlib import Path

# ── Paths ───────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
RAW_DATA    = BASE_DIR / "data"    / "raw"       / "human_cognitive_performance.csv"
PROC_DATA   = BASE_DIR / "data"    / "processed" / "processed_data.csv"
MODELS_DIR  = BASE_DIR / "models"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
REPORTS_DIR = BASE_DIR / "reports"

# Auto-create output directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
(BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

# ── Model constants ─────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2      # 80 / 20 train-test split
N_TRIALS     = 50      # Optuna HPO trials
CV_FOLDS     = 3        # Stratified k-fold