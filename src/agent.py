
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import tool
from langchain.agents import create_agent



embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory="vectordb",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})



def load_resume_text():
    loader = PyPDFLoader(
        "/Users/sahibjotsingh/Desktop/resume/Sahibjot_Resume.pdf"
    )
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)


resume_text = load_resume_text()

@tool
def resume_reader(_: str) -> str:
    """Returns the full text of the user's resume."""
    return resume_text


@tool
def job_search_tool(query: str) -> str:
    """Search job descriptions in vector DB and return relevant job content."""
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)


tools = [resume_reader, job_search_tool]



llm = ChatOllama(
    model="mistral",   
    temperature=0
)


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are a strict career analysis agent.

RULES:
- You MUST call resume_reader tool before analyzing resume.
- You MUST call job_search_tool to retrieve Data Engineer requirements.
- You are NOT allowed to assume resume content.
- You are NOT allowed to fabricate names or information.
- You must use tool outputs only.
- If tools are not used, the answer is invalid.

Steps:
1. Call resume_reader.
2. Call job_search_tool with 'Data Engineer'.
3. Compare actual content.
4. List missing skills.
5. Generate 4-week roadmap.
"""
)




response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze my resume and tell me what skills I am missing for Data Engineer roles."
            }
        ]
    }
)

print("Final Answer:")

print(response["messages"][-1].content)
