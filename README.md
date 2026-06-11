# 🧠 Resume AI — Career Advisor

An AI-powered career advisor that compares your resume against **live job postings** across LinkedIn, Indeed, Glassdoor, and more. It identifies your skill gaps, generates a personalised 4-week learning roadmap, and finds real companies currently hiring for your target role — with direct apply links.

Built with **Google Gemini**, **LangGraph**, **JSearch (RapidAPI)**, **FastAPI**, and **React + Tailwind CSS**.

---

## Features

- **Skill Gap Analysis** — Upload your resume and a target role; the AI agent fetches live job descriptions and tells you exactly what skills you're missing.
- **4-Week Roadmap** — Actionable week-by-week learning plan tailored to your specific gaps.
- **Live Job Finder** — Searches real job postings in real time with filters for job type, experience level, and location.
- **Job Cards with Apply Links** — Company logo, title, location, remote tag, posted date, and a direct Apply Now button.
- **Two AI Agents** — A dedicated Skill Gap agent and a separate Job Finder agent, each with their own tools and prompts.
- **Professional UI** — React + Tailwind CSS frontend with drag-and-drop resume upload, dark theme, and responsive layout.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini (`gemma-4-31b-it`) via `langchain-google-genai` |
| Agent framework | LangGraph (`create_react_agent`) |
| Job data | JSearch API (RapidAPI) — aggregates LinkedIn, Indeed, Glassdoor |
| PDF parsing | PyPDF + LangChain `PyPDFLoader` |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 + Vite + Tailwind CSS |
| Markdown rendering | `react-markdown` |
| File upload | `react-dropzone` |

---

## Project Structure

```
resume-ai-agent-main/
├── api/
│   └── main.py                  # FastAPI backend — /api/analyze and /api/jobs
├── src/
│   ├── agent.py                 # Skill Gap Analysis agent
│   ├── job_finder_agent.py      # Job Finder agent
│   ├── tools.py                 # Shared tools (resume reader, job search)
│   └── config.py                # Environment variable loading
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Root layout, tabs, shared inputs
│   │   ├── pages/
│   │   │   ├── SkillGap.jsx     # Skill gap analysis page
│   │   │   └── JobFinder.jsx    # Job finder page with filters
│   │   └── components/
│   │       ├── ResumeUpload.jsx # Drag-and-drop PDF uploader
│   │       ├── JobCard.jsx      # Individual job posting card
│   │       └── AnalysisResult.jsx # Markdown renderer
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── data/                        # (git-ignored) job dataset CSV
├── vectordb/                    # (git-ignored) Chroma vector DB
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
└── README.md
```

---

## Prerequisites

- Python 3.12
- Node.js 18+
- A **Google AI Studio** API key → [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- A **JSearch RapidAPI** key (free tier) → [rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch)

---

## Setup

### 1. Clone the repo

```bash
git clone <repository_url>
cd resume-ai-agent-main
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
JSEARCH_API_KEY=your_jsearch_rapidapi_key_here
LLM_MODEL=gemma-4-31b-it

# CLI only — not needed for the web UI
RESUME_PATH=/path/to/your/resume.pdf
```

### 5. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running Locally

Start both servers (in two separate terminals):

**Terminal 1 — Backend:**
```bash
./venv/bin/uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### CLI usage (optional)

If you prefer the command line and have `RESUME_PATH` set in `.env`:

```bash
./venv/bin/python -m src.agent --role "Data Engineer"
```

---

## Deployment

### Backend → Render

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables: `GOOGLE_API_KEY`, `JSEARCH_API_KEY`, `LLM_MODEL`

### Frontend → Vercel

1. Create a new project on [vercel.com](https://vercel.com)
2. Set **Root directory** to `frontend`
3. Add environment variable: `VITE_API_URL=https://your-render-app.onrender.com`
4. Update API calls in `SkillGap.jsx` and `JobFinder.jsx` to use `import.meta.env.VITE_API_URL`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/analyze` | Skill gap analysis — accepts `resume` (PDF) + `role` (string) |
| `POST` | `/api/jobs` | Job finder — accepts `resume`, `role`, `employment_type`, `location`, `experience_range` |

---

## Security

- **Never commit your `.env` file** — it is git-ignored by default
- Use `.env.example` as a template (contains only placeholder values)
- Rotate your API keys immediately if they are ever exposed
