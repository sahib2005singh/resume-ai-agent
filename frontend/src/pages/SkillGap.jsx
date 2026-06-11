import { useState } from 'react'
import axios from 'axios'
import { Sparkles, AlertCircle, Loader2 } from 'lucide-react'
import AnalysisResult from '../components/AnalysisResult'

const API_BASE = 'https://resume-ai-agent-amb6.onrender.com'

export default function SkillGap({ resumeFile, role }) {
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)

  async function handleAnalyze() {
    if (!resumeFile) return setError('Please upload your resume first.')
    if (!role.trim()) return setError('Please enter a target job role.')
    setError(null)
    setResult(null)
    setLoading(true)

    try {
      const fd = new FormData()
      fd.append('resume', resumeFile)
      fd.append('role', role.trim())

      const { data } = await axios.post(`${API_BASE}/api/analyze`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResult(data.analysis)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <div className="rounded-2xl bg-slate-900 border border-slate-800 p-6">
        <h2 className="font-semibold text-lg mb-1">Skill Gap Analysis</h2>
        <p className="text-slate-400 text-sm">
          The AI agent reads your resume, fetches <span className="text-brand-300">live job postings</span> for
          your target role, and produces a detailed skills comparison with a personalised 4-week roadmap.
        </p>

        {error && (
          <div className="mt-4 flex items-start gap-2 rounded-xl bg-red-950/40
                          border border-red-800 px-4 py-3 text-sm text-red-300">
            <AlertCircle size={15} className="mt-0.5 shrink-0" />
            {error}
          </div>
        )}

        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="mt-5 flex items-center gap-2 px-6 py-3 rounded-xl
                     bg-brand-600 hover:bg-brand-500 disabled:opacity-50
                     disabled:cursor-not-allowed text-white font-semibold text-sm
                     transition-colors duration-150"
        >
          {loading
            ? <><Loader2 size={15} className="animate-spin" /> Analyzing…</>
            : <><Sparkles size={15} /> Analyze My Resume</>
          }
        </button>
      </div>

      {loading && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-8 space-y-4 animate-pulse">
          <div className="h-4 bg-slate-800 rounded w-1/3" />
          <div className="h-3 bg-slate-800 rounded w-full" />
          <div className="h-3 bg-slate-800 rounded w-5/6" />
          <div className="h-3 bg-slate-800 rounded w-4/6" />
          <div className="h-3 bg-slate-800 rounded w-full" />
          <div className="h-3 bg-slate-800 rounded w-3/4" />
        </div>
      )}

      {result && !loading && (
        <div className="rounded-2xl bg-slate-900 border border-slate-800 p-8">
          <div className="flex items-center gap-2 mb-6">
            <Sparkles size={16} className="text-brand-400" />
            <h3 className="font-semibold text-slate-100">Analysis Results</h3>
          </div>
          <AnalysisResult content={result} />
        </div>
      )}
    </div>
  )
}
