import argparse
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.config import LLM_MODEL, GOOGLE_API_KEY, RESUME_PATH
from src.tools import make_tools, load_resume_from_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_analysis(resume_text: str, target_role: str) -> str:
    """
    Core analysis function. Takes resume text and target role,
    returns the final analysis as a string.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    tools = make_tools(resume_text)

    system_prompt = f"""You are a strict career analysis agent.

RULES:
- You MUST call resume_reader tool before analyzing the resume.
- You MUST call job_search_tool to retrieve live {target_role} job postings.
- You are NOT allowed to assume resume content.
- You are NOT allowed to fabricate names or information.
- You must use tool outputs only.

Steps:
1. Call resume_reader.
2. Call job_search_tool with '{target_role}'.
3. Compare actual resume content against job requirements.
4. List the missing skills clearly.
5. Generate a 4-week learning roadmap to close the skill gaps.
"""

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    response = agent.invoke({
        "messages": [
            HumanMessage(
                content=f"Analyze my resume and tell me what skills I am missing for {target_role} roles."
            )
        ]
    })

    raw = response["messages"][-1].content
    if isinstance(raw, list):
        return "\n".join(
            block["text"] for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return raw


# --- CLI entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume AI Agent for Skill Gap Analysis")
    parser.add_argument("--role", type=str, default="Data Engineer",
                        help="Target job role (e.g., 'Data Engineer')")
    args = parser.parse_args()

    if not RESUME_PATH:
        raise ValueError("Set RESUME_PATH in .env to use the CLI.")

    resume_text = load_resume_from_path(RESUME_PATH)
    result = run_analysis(resume_text, args.role)
    print("\n--- Final Answer ---\n")
    print(result)
