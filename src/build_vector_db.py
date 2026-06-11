import os
import logging
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.config import JOB_DATASET_PATH, VECTORDB_DIR, EMBEDDING_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info(f"Checking dataset path: {JOB_DATASET_PATH}")
    if not os.path.exists(JOB_DATASET_PATH):
        logger.error(f"Dataset not found at {JOB_DATASET_PATH}. Please ensure it exists.")
        return

    logger.info("Loading dataset...")
    try:
        df = pd.read_csv(JOB_DATASET_PATH)
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        return
    
    documents = []
    for _, row in df.iterrows():
        text = str(row.get("Responsibilities", ""))
        if not text or text.lower() == "nan":
            continue
        metadata = {
            "title": str(row.get("Title", "")),
            "skills": str(row.get("Skills", ""))
        }
        documents.append(Document(page_content=text, metadata=metadata))

    logger.info(f"Loaded {len(documents)} documents. Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(documents)
    
    logger.info("Generating embeddings and building Chroma DB...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=str(VECTORDB_DIR)
        )
        logger.info("Successfully built vector DB!")
    except Exception as e:
        logger.error(f"Failed to build vector DB: {e}")

if __name__ == "__main__":
    main()

