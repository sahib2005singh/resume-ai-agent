import { useState } from 'react'
import { BrainCircuit } from 'lucide-react'
import SkillGap from './pages/SkillGap'
import JobFinder from './pages/JobFinder'
import ResumeUpload from './components/ResumeUpload'

const TABS = ['Skill Gap Analysis', 'Job Finder']

export default function App() {
  const [activeTab, setActiveTab] = useState(0)
  const [resumeFile, setResumeFile]   = useState(null)
  const [role, setRole]               = useState('')

  return (
    <div className="min-h-screen flex flex-col">

      {/* ── Nav ─────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center gap-3">
          <BrainCircuit className="text-brand-400" size={26} />
          <span className="font-bold text-lg tracking-tight">
            Resume<span className="text-brand-400">AI</span>
          </span>
          <span className="ml-2 text-xs font-medium px-2 py-0.5 rounded-full bg-brand-900 text-brand-300 border border-brand-700">
            Beta
          </span>
        </div>
      </header>

      {/* ── Hero ────────────────────────────────────────────────────── */}
      <section className="bg-gradient-to-b from-slate-900 to-slate-950 border-b border-slate-800">
        <div className="max-w-6xl mx-auto px-6 py-14">
          <h1 className="text-4xl font-extrabold tracking-tight mb-3">
            AI-Powered Career Advisor
          </h1>
          <p className="text-slate-400 text-lg max-w-xl">
            Upload your resume and discover your skill gaps against live job
            postings — or find companies hiring for your target role right now.
          </p>

          {/* Shared inputs */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
            <ResumeUpload file={resumeFile} onFile={setResumeFile} />

            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Target Role
              </label>
              <input
                type="text"
                value={role}
                onChange={e => setRole(e.target.value)}
                placeholder="e.g. Data Engineer"
                className="h-12 px-4 rounded-xl bg-slate-800 border border-slate-700
                           text-sm text-slate-100 placeholder-slate-500
                           focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent
                           transition"
              />
            </div>
          </div>
        </div>
      </section>

      {/* ── Tabs ────────────────────────────────────────────────────── */}
      <div className="border-b border-slate-800 bg-slate-950">
        <div className="max-w-6xl mx-auto px-6 flex gap-1">
          {TABS.map((tab, i) => (
            <button
              key={tab}
              onClick={() => setActiveTab(i)}
              className={`px-5 py-3.5 text-sm font-medium border-b-2 transition-colors
                ${activeTab === i
                  ? 'border-brand-400 text-brand-300'
                  : 'border-transparent text-slate-500 hover:text-slate-300'}`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* ── Page content ────────────────────────────────────────────── */}
      <main className="flex-1 max-w-6xl mx-auto px-6 py-10 w-full">
        {activeTab === 0
          ? <SkillGap  resumeFile={resumeFile} role={role} />
          : <JobFinder resumeFile={resumeFile} role={role} />
        }
      </main>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-600">
        Powered by Google Gemini · JSearch (RapidAPI) · LangGraph
      </footer>
    </div>
  )
}
