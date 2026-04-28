from pathlib import Path

#  SETTINGS 

# List of tickers used in the portfolio
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB","T", "GS"]

# global time period for the project
START_DATE = "2013-01-01"   # start of historical data
END_DATE = "2018-03-01"     # end of historical data

# train / test split ratio 
TEST_RATIO = 0.3

#  random seed
SEED = 42

# PATHS 


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR = DATA_DIR / "raw"
REPORT_FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

MODELS_DIR = PROJECT_ROOT / "models"
PPO_MODEL_PATH = MODELS_DIR / "ppo_portfolio"
