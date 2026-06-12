import { ArrowRight, BrainCircuit, Search, FileText, Map } from 'lucide-react'

const FEATURES = [
  {
    icon: FileText,
    label: 'Resume Intelligence',
    title: 'Your Resume, Decoded',
    body: 'Upload any PDF resume. The agent reads every word — skills, experience, projects, education — and builds a precise profile of where you stand today.',
  },
  {
    icon: Search,
    label: 'Live Job Market',
    title: 'Real Postings, Right Now',
    body: 'Pulls live job listings from LinkedIn, Indeed, Glassdoor, and more via JSearch. No static databases. No outdated requirements. Only what employers are asking for today.',
  },
  {
    icon: BrainCircuit,
    label: 'AI Gap Analysis',
    title: 'The Gap, Identified',
    body: 'Gemini AI compares your actual skills against real job requirements — not generic checklists. Produces a precise skill-gap matrix showing exactly what is missing and why.',
  },
  {
    icon: Map,
    label: 'Learning Roadmap',
    title: 'Four Weeks to Ready',
    body: 'A week-by-week action plan tailored to your specific gaps. Not generic courses — a structured sequence built around what the market is hiring for right now.',
  },
]

const STATS = [
  { value: '10+', label: 'Job Platforms Scanned' },
  { value: 'Live', label: 'Real-Time Data' },
  { value: 'AI', label: 'Gemini Powered' },
  { value: '4-Week', label: 'Actionable Roadmap' },
]

export default function Landing({ onStart }) {
  return (
    <div className="bg-canvas min-h-screen">

      {/* ── Nav ─────────────────────────────────────────────────────── */}
      <nav className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center justify-between px-8 hairline-bottom" style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)' }}>
        <span className="font-mono text-xs tracking-caption text-muted uppercase">Career Advisor</span>
        <span className="font-display text-sm tracking-wordmark text-ink uppercase">ResumeAI</span>
        <button onClick={onStart} className="font-mono text-xs tracking-caption text-muted uppercase hover:text-ink transition-colors">
          Launch →
        </button>
      </nav>

      {/* ── Hero ────────────────────────────────────────────────────── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-8 text-center"
        style={{ background: 'radial-gradient(ellipse 80% 60% at 50% 0%, #1a1a1a 0%, #000000 70%)' }}>

        {/* Decorative grid lines */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute left-1/4 top-0 bottom-0 w-px" style={{ background: 'linear-gradient(to bottom, transparent, #262626 20%, #262626 80%, transparent)' }} />
          <div className="absolute right-1/4 top-0 bottom-0 w-px" style={{ background: 'linear-gradient(to bottom, transparent, #262626 20%, #262626 80%, transparent)' }} />
        </div>

        <div className="relative max-w-4xl mx-auto space-y-8">
          <p className="caption-label tracking-caption mb-6">AI-Powered Career Intelligence</p>

          <h1 className="font-display text-white uppercase leading-none"
            style={{ fontSize: 'clamp(40px, 8vw, 88px)', letterSpacing: '4px', fontWeight: 400 }}>
            Know Exactly<br />
            <span style={{ color: '#c3d9f3' }}>What's Missing</span>
          </h1>

          <p className="font-body text-body text-lg max-w-xl mx-auto leading-relaxed" style={{ letterSpacing: 0 }}>
            Upload your resume. The AI agent scans live job postings across every major platform
            and tells you precisely which skills are standing between you and your target role.
          </p>

          <div className="flex items-center justify-center gap-4 pt-4">
            <button onClick={onStart} className="btn-primary">
              Analyze My Resume <ArrowRight size={14} />
            </button>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2">
          <span className="caption-label" style={{ fontSize: '10px' }}>Scroll</span>
          <div className="w-px h-12" style={{ background: 'linear-gradient(to bottom, #262626, transparent)' }} />
        </div>
      </section>

      {/* ── Stats band ──────────────────────────────────────────────── */}
      <section className="hairline-top hairline-bottom" style={{ background: '#0d0d0d' }}>
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4">
          {STATS.map((s, i) => (
            <div key={i} className={`px-8 py-10 text-center ${i < 3 ? 'border-r border-[#262626]' : ''}`}>
              <div className="font-display text-white uppercase" style={{ fontSize: 32, letterSpacing: '2px' }}>{s.value}</div>
              <div className="caption-label mt-2">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── How it works ────────────────────────────────────────────── */}
      <section style={{ padding: '120px 32px' }}>
        <div className="max-w-5xl mx-auto">
          <p className="caption-label mb-6">How It Works</p>
          <h2 className="font-display text-white uppercase mb-16"
            style={{ fontSize: 'clamp(28px, 4vw, 48px)', letterSpacing: '3px' }}>
            Four Steps.<br />One Clear Path.
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-px" style={{ background: '#262626' }}>
            {FEATURES.map((f, i) => {
              const Icon = f.icon
              return (
                <div key={i} className="p-10 space-y-4" style={{ background: '#000000' }}>
                  <div className="flex items-center gap-3 mb-6">
                    <span className="caption-label text-muted-soft" style={{ fontSize: 11 }}>0{i + 1}</span>
                    <div className="flex-1 h-px" style={{ background: '#262626' }} />
                  </div>
                  <p className="caption-label" style={{ color: '#c3d9f3' }}>{f.label}</p>
                  <h3 className="font-display text-white uppercase"
                    style={{ fontSize: 24, letterSpacing: '1.5px' }}>
                    {f.title}
                  </h3>
                  <p className="font-body text-body text-sm leading-relaxed" style={{ letterSpacing: 0, fontSize: 15 }}>
                    {f.body}
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* ── Two agents callout ──────────────────────────────────────── */}
      <section className="hairline-top" style={{ padding: '120px 32px', background: '#0d0d0d' }}>
        <div className="max-w-5xl mx-auto">
          <p className="caption-label mb-6">Two Dedicated Agents</p>
          <h2 className="font-display text-white uppercase mb-16"
            style={{ fontSize: 'clamp(28px, 4vw, 48px)', letterSpacing: '3px' }}>
            Built for Precision.
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

            <div className="p-8 space-y-4 hairline-top">
              <p className="caption-label" style={{ color: '#c3d9f3' }}>Agent 01</p>
              <h3 className="font-display text-white uppercase" style={{ fontSize: 28, letterSpacing: '2px' }}>
                Skill Gap Analyst
              </h3>
              <p className="font-body text-body leading-relaxed" style={{ fontSize: 15, letterSpacing: 0 }}>
                Reads your resume line by line. Fetches live job descriptions for your target role.
                Produces a precise comparison table of skills you have versus what employers require —
                and a four-week roadmap to close the gap.
              </p>
              <ul className="space-y-2 pt-2">
                {['Live JD Analysis', 'Skill Matrix Comparison', '4-Week Learning Plan'].map(item => (
                  <li key={item} className="flex items-center gap-3">
                    <span style={{ width: 4, height: 4, background: '#c3d9f3', borderRadius: 0, display: 'inline-block', flexShrink: 0 }} />
                    <span className="caption-label" style={{ color: '#999999' }}>{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="p-8 space-y-4 hairline-top">
              <p className="caption-label" style={{ color: '#c3d9f3' }}>Agent 02</p>
              <h3 className="font-display text-white uppercase" style={{ fontSize: 28, letterSpacing: '2px' }}>
                Job Finder
              </h3>
              <p className="font-body text-body leading-relaxed" style={{ fontSize: 15, letterSpacing: 0 }}>
                Scans LinkedIn, Indeed, Glassdoor, and more in real time. Filters by job type,
                experience level, and location. Returns company cards with logos, descriptions,
                and direct apply links — ranked for your profile.
              </p>
              <ul className="space-y-2 pt-2">
                {['Real-Time Job Listings', 'Filter by Type & Location', 'Direct Apply Links'].map(item => (
                  <li key={item} className="flex items-center gap-3">
                    <span style={{ width: 4, height: 4, background: '#c3d9f3', borderRadius: 0, display: 'inline-block', flexShrink: 0 }} />
                    <span className="caption-label" style={{ color: '#999999' }}>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA band ────────────────────────────────────────────────── */}
      <section className="hairline-top text-center" style={{ padding: '120px 32px' }}>
        <div className="max-w-2xl mx-auto space-y-8">
          <p className="caption-label">Get Started</p>
          <h2 className="font-display text-white uppercase"
            style={{ fontSize: 'clamp(32px, 5vw, 56px)', letterSpacing: '3px' }}>
            Your Next Role<br />Starts Here.
          </h2>
          <p className="font-body text-body" style={{ fontSize: 15, letterSpacing: 0 }}>
            Upload your resume. Enter your target role. The agents do the rest.
          </p>
          <button onClick={onStart} className="btn-primary mx-auto">
            Begin Analysis <ArrowRight size={14} />
          </button>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="hairline-top px-8 py-16">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <span className="font-display tracking-wordmark text-ink uppercase text-sm">ResumeAI</span>
          <p className="caption-label text-center" style={{ color: '#666666' }}>
            Powered by Google Gemini · JSearch (RapidAPI) · LangGraph
          </p>
          <p className="caption-label" style={{ color: '#666666', fontSize: 10 }}>
            © 2026 ResumeAI
          </p>
        </div>
      </footer>
    </div>
  )
}
