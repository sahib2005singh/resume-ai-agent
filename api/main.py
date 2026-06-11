import tempfile
import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.tools import load_resume_from_path
from src.agent import run_analysis
from src.job_finder_agent import run_job_finder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Resume AI Career Advisor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ──────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    analysis: str

class JobItem(BaseModel):
    title: str
    company: str
    location: str
    apply_link: str
    logo: str | None = None
    posted: str
    employment_type: str
    is_remote: bool
    description: str

class JobFinderResponse(BaseModel):
    summary: str
    jobs: list[JobItem]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    resume: UploadFile = File(..., description="PDF resume"),
    role: str = Form(..., description="Target job role"),
):
    """Skill Gap Analysis Agent — compares resume against live job postings."""
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await resume.read())
            tmp_path = tmp.name

        resume_text = load_resume_from_path(tmp_path)
        if resume_text.startswith("Error"):
            raise HTTPException(status_code=422, detail=resume_text)

        result = run_analysis(resume_text, role)
        return AnalysisResponse(analysis=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs", response_model=JobFinderResponse)
async def find_jobs(
    resume: UploadFile = File(..., description="PDF resume"),
    role: str          = Form(...,  description="Target job role"),
    employment_type: str = Form("FULLTIME", description="FULLTIME | INTERN | PARTTIME | CONTRACTOR"),
    location: str        = Form("",         description="Preferred location (city, country)"),
    experience_range: str = Form("",        description="e.g. 0-2 | 2-5 | 5+"),
):
    """Job Finder Agent — finds live openings with company cards and apply links."""
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await resume.read())
            tmp_path = tmp.name

        resume_text = load_resume_from_path(tmp_path)
        if resume_text.startswith("Error"):
            raise HTTPException(status_code=422, detail=resume_text)

        ai_summary, job_list = run_job_finder(
            resume_text=resume_text,
            target_role=role,
            employment_type=employment_type,
            location=location,
            experience_range=experience_range,
        )

        jobs = [JobItem(**j) for j in job_list]
        return JobFinderResponse(summary=ai_summary, jobs=jobs)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job finder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
