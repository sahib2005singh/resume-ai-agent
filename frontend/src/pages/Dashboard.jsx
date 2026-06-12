import { useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import ResumeUpload from '../components/ResumeUpload'
import SkillGap from './SkillGap'
import JobFinder from './JobFinder'

const TABS = [
  { id: 'skill', label: 'Skill Gap Analysis' },
  { id: 'jobs',  label: 'Job Finder' },
]

export default function Dashboard() {
  const [activeTab,   setActiveTab]   = useState('skill')
  const [resumeFile,  setResumeFile]  = useState(null)
  const [role,        setRole]        = useState('')

  return (
    <div className="min-h-screen bg-canvas flex flex-col">

      {/* ── Nav ──────────────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 h-14 flex items-center justify-between px-8 hairline-bottom"
        style={{ background: 'rgba(0,0,0,0.9)', backdropFilter: 'blur(12px)' }}>
        <button
          onClick={() => window.location.reload()}
          className="flex items-center gap-2 font-mono text-xs tracking-caption text-muted uppercase hover:text-ink transition-colors"
        >
          <ArrowLeft size={12} /> Back
        </button>
        <span className="font-display text-sm tracking-wordmark text-ink uppercase">ResumeAI</span>
        <span className="caption-label" style={{ color: '#666666' }}>Career Advisor</span>
      </nav>

      {/* ── Inputs band ──────────────────────────────────────────── */}
      <div className="hairline-bottom" style={{ background: '#0d0d0d', padding: '40px 32px' }}>
        <div className="max-w-4xl mx-auto">
          <p className="caption-label mb-6">Your Profile</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div>
              <label className="caption-label block mb-3">Resume (PDF)</label>
              <ResumeUpload file={resumeFile} onFile={setResumeFile} />
            </div>
            <div>
              <label className="caption-label block mb-3">Target Role</label>
              <input
                type="text"
                value={role}
                onChange={e => setRole(e.target.value)}
                placeholder="e.g. Data Engineer"
                className="input-bugatti"
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── Tabs ─────────────────────────────────────────────────── */}
      <div className="hairline-bottom">
        <div className="max-w-4xl mx-auto px-8 flex">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className="font-mono text-xs uppercase py-4 pr-8 transition-colors"
              style={{
                letterSpacing: '2px',
                color: activeTab === tab.id ? '#ffffff' : '#666666',
                borderBottom: activeTab === tab.id ? '1px solid #ffffff' : '1px solid transparent',
                marginBottom: '-1px',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Content ──────────────────────────────────────────────── */}
      <main className="flex-1 max-w-4xl mx-auto w-full px-8 py-16">
        {activeTab === 'skill'
          ? <SkillGap  resumeFile={resumeFile} role={role} />
          : <JobFinder resumeFile={resumeFile} role={role} />
        }
      </main>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <footer className="hairline-top px-8 py-8 text-center">
        <p className="caption-label" style={{ color: '#444444', fontSize: 10 }}>
          Powered by Google Gemini · JSearch (RapidAPI) · LangGraph
        </p>
      </footer>
    </div>
  )
}
