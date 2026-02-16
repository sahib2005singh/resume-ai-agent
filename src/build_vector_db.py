import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from langchain_chroma import Chroma

df = pd.read_csv("/Users/sahibjotsingh/resume_ai_agent/data/job_dataset.csv")
documents = []
for _, row in df.iterrows():
    text = str(row["Responsibilities"])
    documents.append(Document(page_content=text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50

)
split_docs = splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding = embeddings,
    persist_directory="vectordb"
)


print("successfull")
