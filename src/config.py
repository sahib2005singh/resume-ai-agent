import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORDB_DIR = BASE_DIR / "vectordb"

# Ensure directories exist
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Resume path — optional, used by CLI only
RESUME_PATH = os.getenv("RESUME_PATH")

JOB_DATASET_PATH = os.getenv("JOB_DATASET_PATH", str(DATA_DIR / "job_dataset.csv"))
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-31b-it")

# JSearch (RapidAPI) Key
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")
if not JSEARCH_API_KEY:
    raise ValueError("JSEARCH_API_KEY is not set. Please add it to your .env file.")

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
