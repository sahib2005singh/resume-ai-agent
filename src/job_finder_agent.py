import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import requests
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.config import LLM_MODEL, GOOGLE_API_KEY, JSEARCH_API_KEY

logger = logging.getLogger(__name__)

JSEARCH_URL = "https://jsearch.p.rapidapi.com/search"
JSEARCH_HEADERS = {
    "X-RapidAPI-Key": JSEARCH_API_KEY,
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

EMPLOYMENT_TYPE_MAP = {
    "internship": "INTERN",
    "intern":     "INTERN",
    "fulltime":   "FULLTIME",
    "full-time":  "FULLTIME",
    "full_time":  "FULLTIME",
    "parttime":   "PARTTIME",
    "part-time":  "PARTTIME",
    "contract":   "CONTRACTOR",
}


def fetch_job_openings(
    role: str,
    employment_type: str = "FULLTIME",
    location: str = "",
    experience_range: str = "",
    num_results: int = 10
) -> list[dict]:
    """
    Calls JSearch and returns structured job opening data.
    Filters by employment type, location, and experience range (via query enrichment).
    """
    try:
        # Enrich the query with experience + location context
        query_parts = [role]
        if experience_range:
            query_parts.append(f"{experience_range} years experience")
        if location:
            query_parts.append(location)
        query = " ".join(query_parts)

        params = {
            "query": query,
            "page": "1",
            "num_pages": "1",
            "date_posted": "month",
            "employment_types": employment_type,
        }
        if location:
            params["location"] = location

        response = requests.get(
            JSEARCH_URL, headers=JSEARCH_HEADERS, params=params, timeout=15
        )
        response.raise_for_status()
        jobs = response.json().get("data", [])

        results = []
        for job in jobs[:num_results]:
            city    = job.get("job_city") or ""
            state   = job.get("job_state") or ""
            country = job.get("job_country") or ""
            loc     = ", ".join(filter(None, [city, state, country])) or "Remote / Not specified"

            results.append({
                "title":           job.get("job_title", "N/A"),
                "company":         job.get("employer_name", "N/A"),
                "location":        loc,
                "apply_link":      job.get("job_apply_link", ""),
                "logo":            job.get("employer_logo", ""),
                "posted":          job.get("job_posted_at_datetime_utc", "N/A"),
                "employment_type": job.get("job_employment_type", employment_type),
                "is_remote":       job.get("job_is_remote", False),
                "description":     (job.get("job_description") or "")[:300],
            })

        return results

    except requests.exceptions.HTTPError as e:
        logger.error(f"JSearch HTTP error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching job openings: {e}")
        return []


def make_job_finder_tools(
    resume_text: str,
    employment_type: str,
    location: str,
    experience_range: str,
) -> list:
    """Build tools for the job finder agent, with search params baked in."""

    @tool
    def resume_reader(query: str = "") -> str:
        """Returns the full text of the user's resume. No input needed."""
        return resume_text

    @tool
    def find_job_openings(role: str) -> str:
        """
        Searches for live job openings for the given role across LinkedIn,
        Indeed, Glassdoor, and more. Filters by employment type, location,
        and experience level. Returns company names, titles, locations, and apply links.
        """
        jobs = fetch_job_openings(
            role=role,
            employment_type=employment_type,
            location=location,
            experience_range=experience_range,
            num_results=10,
        )
        if not jobs:
            return f"No job openings found for '{role}' with the given filters."

        lines = [f"Found {len(jobs)} live openings for '{role}':\n"]
        for i, job in enumerate(jobs, 1):
            remote_tag = " [REMOTE]" if job["is_remote"] else ""
            lines.append(
                f"{i}. {job['title']} at {job['company']}\n"
                f"   Location: {job['location']}{remote_tag}\n"
                f"   Type: {job['employment_type']}\n"
                f"   Posted: {job['posted']}\n"
                f"   Apply: {job['apply_link']}\n"
            )
        return "\n".join(lines)

    return [resume_reader, find_job_openings]


def run_job_finder(
    resume_text: str,
    target_role: str,
    employment_type: str = "FULLTIME",
    location: str = "",
    experience_range: str = "",
) -> tuple[str, list[dict]]:
    """
    Job Finder Agent.
    Returns (ai_summary, job_list).
    """
    # Normalise employment type
    emp_type = EMPLOYMENT_TYPE_MAP.get(employment_type.lower().replace(" ", ""), "FULLTIME")

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    tools = make_job_finder_tools(resume_text, emp_type, location, experience_range)

    filters_desc = []
    if employment_type:
        filters_desc.append(f"type: {employment_type}")
    if location:
        filters_desc.append(f"location: {location}")
    if experience_range:
        filters_desc.append(f"experience: {experience_range} years")
    filters_str = " | ".join(filters_desc) if filters_desc else "no specific filters"

    system_prompt = f"""You are a job matching agent helping a candidate find '{target_role}' openings.
Search filters applied: {filters_str}

RULES:
- You MUST call resume_reader first to understand the candidate's background.
- You MUST call find_job_openings with '{target_role}' to get live openings.
- Do NOT fabricate company names, links, or job details.
- Only use data returned by the tools.

Output format:
1. A brief 2-3 sentence summary of the candidate's profile fit for this role.
2. Top 5 recommended companies from the results with a one-line reason why each is a good fit.
3. Any patterns you notice (e.g., most roles require X, many are remote, common skills asked).
"""

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    response = agent.invoke({
        "messages": [
            HumanMessage(
                content=f"Find the best '{target_role}' openings for me based on my resume. Filters: {filters_str}"
            )
        ]
    })

    raw = response["messages"][-1].content
    if isinstance(raw, list):
        ai_summary = "\n".join(
            block["text"] for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        ai_summary = raw

    job_list = fetch_job_openings(
        role=target_role,
        employment_type=emp_type,
        location=location,
        experience_range=experience_range,
        num_results=10,
    )

    return ai_summary, job_list
