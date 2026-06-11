import { useState } from 'react'
import axios from 'axios'
import { Search, AlertCircle, Loader2, BriefcaseBusiness, MapPin } from 'lucide-react'
import JobCard from '../components/JobCard'
import AnalysisResult from '../components/AnalysisResult'

const EMPLOYMENT_TYPES = [
  { label: 'Full-Time',   value: 'FULLTIME'   },
  { label: 'Internship',  value: 'INTERN'     },
  { label: 'Part-Time',   value: 'PARTTIME'   },
  { label: 'Contract',    value: 'CONTRACTOR' },
]

const EXPERIENCE_RANGES = [
  { label: '0 – 2 years',  value: '0-2' },
  { label: '2 – 5 years',  value: '2-5' },
  { label: '5 – 10 years', value: '5-10' },
  { label: '10+ years',    value: '10+' },
]

export default function JobFinder({ resumeFile, role }) {
  const [empType,    setEmpType]    = useState('FULLTIME')
  const [location,   setLocation]   = useState('')
  const [expRange,   setExpRange]   = useState('')
  const [loading,    setLoading]    = useState(false)
  const [summary,    setSummary]    = useState(null)
  const [jobs,       setJobs]       = useState([])
  const [error,      setError]      = useState(null)

  const isIntern = empType === 'INTERN'

  async function handleSearch() {
    if (!resumeFile) return setError('Please upload your resume first.')
    if (!role.trim()) return setError('Please enter a target job role.')
    setError(null)
    setSummary(null)
    setJobs([])
    setLoading(true)

    try {
      const fd = new FormData()
      fd.append('resume', resumeFile)
      fd.append('role', role.trim())
      fd.append('employment_type', empType)
      fd.append('location', location.trim())
      fd.append('experience_range', isIntern ? '' : expRange)

      const { data } = await axios.post('/api/jobs', fd, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setSummary(data.summary)
      setJobs(data.jobs || [])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">

      {/* ── Filter panel ──────────────────────────────────────────── */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-5">
        <div>
          <h2 className="font-semibold text-lg mb-0.5">Job Finder</h2>
          <p className="text-slate-400 text-sm">
            Find companies hiring for your role right now with direct apply links.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">

          {/* Employment type */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <BriefcaseBusiness size={12} /> Job Type
            </label>
            <div className="flex flex-wrap gap-2">
              {EMPLOYMENT_TYPES.map(t => (
                <button
                  key={t.value}
                  onClick={() => { setEmpType(t.value); setExpRange('') }}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition
                    ${empType === t.value
                      ? 'bg-brand-600 border-brand-500 text-white'
                      : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500'}`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* Experience — hidden for intern */}
          {!isIntern && (
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Experience
              </label>
              <div className="flex flex-wrap gap-2">
                {EXPERIENCE_RANGES.map(r => (
                  <button
                    key={r.value}
                    onClick={() => setExpRange(expRange === r.value ? '' : r.value)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition
                      ${expRange === r.value
                        ? 'bg-brand-600 border-brand-500 text-white'
                        : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-500'}`}
                  >
                    {r.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Location */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <MapPin size={12} /> Location
            </label>
            <input
              type="text"
              value={location}
              onChange={e => setLocation(e.target.value)}
              placeholder="e.g. San Francisco, Remote, India"
              className="h-10 px-3 rounded-xl bg-slate-800 border border-slate-700
                         text-sm text-slate-100 placeholder-slate-500
                         focus:outline-none focus:ring-2 focus:ring-brand-500
                         transition"
            />
          </div>
        </div>

        {/* Intern note */}
        {isIntern && (
          <p className="text-xs text-purple-400 bg-purple-950/30 border border-purple-800
                         rounded-xl px-4 py-2.5">
            Internship search — no experience filter applied.
          </p>
        )}

        {error && (
          <div className="flex items-start gap-2 rounded-xl bg-red-950/40
                          border border-red-800 px-4 py-3 text-sm text-red-300">
            <AlertCircle size={15} className="mt-0.5 shrink-0" />
            {error}
          </div>
        )}

        <button
          onClick={handleSearch}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-3 rounded-xl
                     bg-brand-600 hover:bg-brand-500 disabled:opacity-50
                     disabled:cursor-not-allowed text-white font-semibold text-sm
                     transition-colors duration-150"
        >
          {loading
            ? <><Loader2 size={15} className="animate-spin" /> Searching…</>
            : <><Search size={15} /> Find Jobs</>
          }
        </button>
      </div>

      {/* ── Loading skeleton ──────────────────────────────────────── */}
      {loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="rounded-2xl bg-slate-900 border border-slate-800 p-5 animate-pulse space-y-3">
              <div className="flex gap-3">
                <div className="w-12 h-12 rounded-xl bg-slate-800" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-slate-800 rounded w-3/4" />
                  <div className="h-3 bg-slate-800 rounded w-1/2" />
                </div>
              </div>
              <div className="h-3 bg-slate-800 rounded w-full" />
              <div className="h-3 bg-slate-800 rounded w-5/6" />
              <div className="h-10 bg-slate-800 rounded-xl mt-2" />
            </div>
          ))}
        </div>
      )}

      {/* ── Results ───────────────────────────────────────────────── */}
      {!loading && jobs.length > 0 && (
        <>
          {/* AI summary */}
          {summary && (
            <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6">
              <h3 className="font-semibold text-slate-100 mb-4 flex items-center gap-2">
                <Search size={15} className="text-brand-400" />
                AI Match Summary
              </h3>
              <AnalysisResult content={summary} />
            </div>
          )}

          {/* Job cards */}
          <div>
            <h3 className="font-semibold text-slate-100 mb-4">
              {jobs.length} Live Openings for "{role}"
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {jobs.map((job, i) => <JobCard key={i} job={job} />)}
            </div>
          </div>
        </>
      )}

      {/* Empty state */}
      {!loading && jobs.length === 0 && summary && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-10 text-center text-slate-500">
          No openings found. Try a different role or location.
        </div>
      )}
    </div>
  )
}
