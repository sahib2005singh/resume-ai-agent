import { useState } from 'react'
import axios from 'axios'
import { Loader2, RefreshCw } from 'lucide-react'
import JobCard from '../components/JobCard'
import AnalysisResult from '../components/AnalysisResult'

const API_BASE = 'https://resume-ai-agent-amb6.onrender.com'

const EMPLOYMENT_TYPES = [
  { label: 'Full-Time',  value: 'FULLTIME'   },
  { label: 'Internship', value: 'INTERN'     },
  { label: 'Part-Time',  value: 'PARTTIME'   },
  { label: 'Contract',   value: 'CONTRACTOR' },
]

const EXPERIENCE_RANGES = [
  { label: '0 – 2 yrs',  value: '0-2'  },
  { label: '2 – 5 yrs',  value: '2-5'  },
  { label: '5 – 10 yrs', value: '5-10' },
  { label: '10+',        value: '10+'  },
]

function FilterButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className="font-mono text-xs uppercase transition-colors"
      style={{
        letterSpacing: '2px',
        padding: '8px 16px',
        border: `1px solid ${active ? '#ffffff' : '#262626'}`,
        borderRadius: 0,
        background: active ? '#ffffff' : 'transparent',
        color: active ? '#000000' : '#666666',
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}

export default function JobFinder({ resumeFile, role }) {
  const [empType,  setEmpType]  = useState('FULLTIME')
  const [location, setLocation] = useState('')
  const [expRange, setExpRange] = useState('')
  const [loading,  setLoading]  = useState(false)
  const [summary,  setSummary]  = useState(null)
  const [jobs,     setJobs]     = useState([])
  const [error,    setError]    = useState(null)
  const [page,     setPage]     = useState(1)

  const isIntern = empType === 'INTERN'

  async function fetchJobs(pageNum = 1) {
    if (!resumeFile) return setError('Upload your resume first.')
    if (!role.trim()) return setError('Enter a target role.')
    setError(null)
    setLoading(true)
    if (pageNum === 1) { setSummary(null); setJobs([]) }

    try {
      const fd = new FormData()
      fd.append('resume', resumeFile)
      fd.append('role', role.trim())
      fd.append('employment_type', empType)
      fd.append('location', location.trim())
      fd.append('experience_range', isIntern ? '' : expRange)
      fd.append('page', String(pageNum))

      const { data } = await axios.post(`${API_BASE}/api/jobs`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setSummary(data.summary)
      setJobs(data.jobs || [])
      setPage(pageNum)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-16">

      {/* ── Filters ───────────────────────────────────────────── */}
      <div className="space-y-8">
        <p className="font-body text-body leading-relaxed" style={{ fontSize: 15, maxWidth: 540, letterSpacing: 0 }}>
          Find companies hiring for your target role right now — filtered by job type,
          experience level, and location. Each result includes a direct apply link.
        </p>

        {/* Job type */}
        <div className="space-y-3">
          <label className="caption-label">Job Type</label>
          <div className="flex flex-wrap gap-2">
            {EMPLOYMENT_TYPES.map(t => (
              <FilterButton key={t.value} active={empType === t.value}
                onClick={() => { setEmpType(t.value); setExpRange('') }}>
                {t.label}
              </FilterButton>
            ))}
          </div>
        </div>

        {/* Experience */}
        {!isIntern && (
          <div className="space-y-3">
            <label className="caption-label">Experience</label>
            <div className="flex flex-wrap gap-2">
              {EXPERIENCE_RANGES.map(r => (
                <FilterButton key={r.value} active={expRange === r.value}
                  onClick={() => setExpRange(expRange === r.value ? '' : r.value)}>
                  {r.label}
                </FilterButton>
              ))}
            </div>
          </div>
        )}

        {/* Location */}
        <div className="space-y-3" style={{ maxWidth: 320 }}>
          <label className="caption-label">Location</label>
          <input
            type="text"
            value={location}
            onChange={e => setLocation(e.target.value)}
            placeholder="San Francisco, Remote, India…"
            className="input-bugatti"
          />
        </div>

        {isIntern && (
          <p className="caption-label" style={{ color: '#c3d9f3' }}>
            Internship — experience filter not applied
          </p>
        )}

        {error && (
          <p className="font-mono text-xs uppercase tracking-caption" style={{ color: '#d4a017' }}>
            ↳ {error}
          </p>
        )}

        <button onClick={() => fetchJobs(1)} disabled={loading} className="btn-primary">
          {loading
            ? <><Loader2 size={13} className="animate-spin" /> Searching</>
            : 'Find Openings'
          }
        </button>
      </div>

      {/* ── Loading skeleton ──────────────────────────────────── */}
      {loading && (
        <div className="space-y-3">
          <div className="caption-label" style={{ color: '#444' }}>Scanning live job boards…</div>
          {[...Array(4)].map((_, i) => (
            <div key={i} style={{ height: 1, background: '#141414', width: `${[90, 70, 80, 60][i]}%` }} />
          ))}
        </div>
      )}

      {/* ── Results ───────────────────────────────────────────── */}
      {!loading && jobs.length > 0 && (
        <div className="space-y-12">

          {/* AI summary */}
          {summary && (
            <div className="space-y-6">
              <div className="hairline-bottom pb-4">
                <p className="caption-label" style={{ color: '#c3d9f3' }}>AI Match Summary</p>
              </div>
              <AnalysisResult content={summary} />
            </div>
          )}

          {/* Jobs header + refresh */}
          <div className="flex items-center justify-between hairline-bottom pb-4">
            <div>
              <p className="caption-label" style={{ color: '#c3d9f3' }}>Live Openings</p>
              <h2 className="font-display text-ink uppercase mt-1"
                style={{ fontSize: 20, letterSpacing: '1.5px' }}>
                {jobs.length} Results · Page {page}
              </h2>
            </div>
            <button onClick={() => fetchJobs(page + 1)} disabled={loading} className="btn-ghost">
              <RefreshCw size={12} /> Load More
            </button>
          </div>

          {/* Cards grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-px" style={{ background: '#141414' }}>
            {jobs.map((job, i) => <JobCard key={i} job={job} />)}
          </div>
        </div>
      )}

      {!loading && jobs.length === 0 && summary && (
        <p className="caption-label" style={{ color: '#444444' }}>
          No openings found. Try a different role or location.
        </p>
      )}
    </div>
  )
}
