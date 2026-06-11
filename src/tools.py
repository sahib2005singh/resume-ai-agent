import logging
import requests
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from src.config import JSEARCH_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

JSEARCH_URL = "https://jsearch.p.rapidapi.com/search"
JSEARCH_HEADERS = {
    "X-RapidAPI-Key": JSEARCH_API_KEY,
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

def load_resume_from_path(path: str) -> str:
    """Load resume text from a PDF file path."""
    logger.info(f"Loading resume from {path}")
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Error loading resume: {e}")
        return f"Error loading resume: {str(e)}"


def make_tools(resume_text: str) -> list:
    """
    Create agent tools with the resume text baked in.
    Call this once per user session after the resume is uploaded.
    """

    @tool
    def resume_reader(query: str = "") -> str:
        """Returns the full text of the user's resume. No input needed."""
        return resume_text

    @tool
    def job_search_tool(query: str) -> str:
        """
        Fetches live job postings from JSearch (RapidAPI) for the given role/query.
        Returns job titles, companies, and full job descriptions from real postings
        across LinkedIn, Indeed, Glassdoor, and more.
        """
        try:
            params = {
                "query": query,
                "page": "1",
                "num_pages": "1",
                "date_posted": "month",
                "employment_types": "FULLTIME"
            }
            response = requests.get(
                JSEARCH_URL, headers=JSEARCH_HEADERS, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json()

            jobs = data.get("data", [])
            if not jobs:
                return f"No live job postings found for '{query}'."

            results = []
            for job in jobs[:5]:
                title = job.get("job_title", "N/A")
                company = job.get("employer_name", "N/A")
                location = (job.get("job_city", "") + ", " + job.get("job_country", "")).strip(", ")
                description = job.get("job_description", "No description available.")[:1500]
                posted = job.get("job_posted_at_datetime_utc", "N/A")

                results.append(
                    f"--- Job Posting ---\n"
                    f"Title: {title}\n"
                    f"Company: {company}\n"
                    f"Location: {location}\n"
                    f"Posted: {posted}\n"
                    f"Description:\n{description}\n"
                )

            return "\n".join(results)

        except requests.exceptions.HTTPError as e:
            logger.error(f"JSearch HTTP error: {e}")
            return f"Error fetching jobs: HTTP {e.response.status_code} - {e.response.text}"
        except Exception as e:
            logger.error(f"Error fetching live jobs: {e}")
            return f"Error fetching live jobs: {str(e)}"

    return [resume_reader, job_search_tool]
