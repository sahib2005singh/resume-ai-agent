import { useState } from 'react'
import axios from 'axios'
import { Loader2 } from 'lucide-react'
import AnalysisResult from '../components/AnalysisResult'

const API_BASE = 'https://resume-ai-agent-amb6.onrender.com'

export default function SkillGap({ resumeFile, role }) {
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)

  async function handleAnalyze() {
    if (!resumeFile) return setError('Upload your resume first.')
    if (!role.trim()) return setError('Enter a target role.')
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
    <div className="space-y-16">

      {/* Description */}
      <div className="space-y-6">
        <p className="font-body text-body leading-relaxed" style={{ fontSize: 15, maxWidth: 540, letterSpacing: 0 }}>
          The agent reads your resume, fetches live job postings for your target role,
          and produces a precise skill-gap comparison with a four-week learning roadmap.
        </p>

        {error && (
          <p className="font-mono text-xs uppercase tracking-caption" style={{ color: '#d4a017' }}>
            ↳ {error}
          </p>
        )}

        <button onClick={handleAnalyze} disabled={loading} className="btn-primary">
          {loading
            ? <><Loader2 size={13} className="animate-spin" /> Analyzing</>
            : 'Begin Analysis'
          }
        </button>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="space-y-6">
          <div className="caption-label" style={{ color: '#444' }}>
            Reading resume → Fetching live job postings → Analyzing gaps…
          </div>
          <div className="space-y-3">
            {[80, 100, 60, 90, 70].map((w, i) => (
              <div key={i} style={{ height: 1, background: '#141414', width: `${w}%`, animation: `pulse 1.5s ease-in-out ${i * 0.15}s infinite alternate` }} />
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="space-y-8">
          <div className="hairline-bottom pb-4">
            <p className="caption-label" style={{ color: '#c3d9f3' }}>Analysis Complete</p>
            <h2 className="font-display text-white uppercase mt-2"
              style={{ fontSize: 28, letterSpacing: '2px' }}>
              Your Skill Gap Report
            </h2>
          </div>
          <AnalysisResult content={result} />
        </div>
      )}
    </div>
  )
}
