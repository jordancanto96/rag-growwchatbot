from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

CORPUS = [
    {
        "fund_name": "HDFC Large Cap Fund Direct Growth",
        "fund_category": "Large Cap",
        "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    },
    {
        "fund_name": "HDFC Flexi Cap Fund Direct Growth",
        "fund_category": "Flexi Cap",
        "url": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    },
    {
        "fund_name": "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        "fund_category": "ELSS",
        "url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    },
    {
        "fund_name": "HDFC Small Cap Fund Direct Growth",
        "fund_category": "Small Cap",
        "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    },
    {
        "fund_name": "HDFC Balanced Advantage Fund Direct Growth",
        "fund_category": "Hybrid",
        "url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    },
]

CORPUS_URLS = [item["url"] for item in CORPUS]