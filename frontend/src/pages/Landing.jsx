import { ArrowRight } from 'lucide-react'

const FEATURES = [
  {
    number: '01',
    label: 'Resume Intelligence',
    title: 'Your Resume, Decoded',
    body: 'Upload any PDF resume. The agent reads every word — skills, experience, projects, education — and builds a precise profile of where you stand today.',
  },
  {
    number: '02',
    label: 'Live Job Market',
    title: 'Real Postings, Right Now',
    body: 'Pulls live job listings from LinkedIn, Indeed, Glassdoor, and more via JSearch. No static databases. No outdated requirements. Only what employers are asking for today.',
  },
  {
    number: '03',
    label: 'AI Gap Analysis',
    title: 'The Gap, Identified',
    body: 'Gemini AI compares your actual skills against real job requirements — not generic checklists. Produces a precise skill-gap matrix showing exactly what is missing and why.',
  },
  {
    number: '04',
    label: 'Learning Roadmap',
    title: 'Four Weeks to Ready',
    body: 'A week-by-week action plan tailored to your specific gaps. Not generic courses — a structured sequence built around what the market is hiring for right now.',
  },
]

const AGENTS = [
  {
    tag: 'Agent 01',
    title: 'Skill Gap Analyst',
    desc: 'Reads your resume line by line. Fetches live job descriptions for your target role. Produces a precise comparison table of skills you have versus what employers require — and a four-week roadmap to close the gap.',
    items: ['Live JD Analysis', 'Skill Matrix Comparison', '4-Week Learning Plan'],
  },
  {
    tag: 'Agent 02',
    title: 'Job Finder',
    desc: 'Scans LinkedIn, Indeed, Glassdoor, and more in real time. Filters by job type, experience level, and location. Returns company cards with logos, descriptions, and direct apply links.',
    items: ['Real-Time Job Listings', 'Filter by Type & Location', 'Direct Apply Links'],
  },
]

const STATS = [
  { value: '10+',    label: 'Job Platforms' },
  { value: 'Live',   label: 'Real-Time Data' },
  { value: 'Gemini', label: 'AI Engine' },
  { value: '2',      label: 'Dedicated Agents' },
]

export default function Landing({ onStart }) {
  return (
    <div className="bg-canvas" style={{ minHeight: '100vh' }}>
      <style>{`
        .land-nav {
          position: fixed; top: 0; left: 0; right: 0; z-index: 50;
          height: 56px; display: flex; align-items: center;
          justify-content: space-between; padding: 0 24px;
          border-bottom: 1px solid #1a1a1a;
          background: rgba(0,0,0,0.92); backdrop-filter: blur(12px);
        }
        .land-section { max-width: 1200px; margin: 0 auto; padding: 0 24px; }

        /* Hero */
        .hero-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 40px;
          padding-top: 120px;
          padding-bottom: 80px;
        }
        @media (min-width: 768px) {
          .hero-grid {
            grid-template-columns: 1fr 360px;
            gap: 80px;
            padding-top: 160px;
            padding-bottom: 120px;
            align-items: end;
          }
          .land-nav { padding: 0 48px; }
          .land-section { padding: 0 48px; }
        }

        /* Stats */
        .stats-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
        }
        @media (min-width: 768px) {
          .stats-grid { grid-template-columns: repeat(4, 1fr); }
        }
        .stat-cell {
          padding: 28px 20px;
          border-right: 1px solid #1a1a1a;
          border-bottom: 1px solid #1a1a1a;
        }
        .stat-cell:nth-child(2n) { border-right: none; }
        @media (min-width: 768px) {
          .stat-cell { padding: 36px 32px; border-bottom: none; }
          .stat-cell:nth-child(2n) { border-right: 1px solid #1a1a1a; }
          .stat-cell:last-child { border-right: none; }
        }

        /* Section header */
        .section-header {
          display: grid;
          grid-template-columns: 1fr;
          gap: 24px;
          margin-bottom: 60px;
        }
        @media (min-width: 768px) {
          .section-header {
            grid-template-columns: 240px 1fr;
            gap: 80px;
            margin-bottom: 80px;
            align-items: end;
          }
        }

        /* Feature rows */
        .feature-row {
          display: grid;
          grid-template-columns: 40px 1fr;
          gap: 20px;
          padding: 32px 0;
          border-bottom: 1px solid #1a1a1a;
          align-items: start;
        }
        @media (min-width: 768px) {
          .feature-row {
            grid-template-columns: 80px 240px 1fr;
            gap: 48px;
            padding: 40px 0;
          }
          .feature-body { display: block; }
        }
        .feature-main { grid-column: 2; }
        @media (min-width: 768px) {
          .feature-main { grid-column: auto; }
        }

        /* Agents */
        .agents-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 1px;
          background: #1a1a1a;
        }
        @media (min-width: 768px) {
          .agents-grid { grid-template-columns: 1fr 1fr; }
        }

        /* CTA */
        .cta-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 40px;
          padding: 80px 0;
        }
        @media (min-width: 768px) {
          .cta-grid {
            grid-template-columns: 1fr 1fr;
            gap: 80px;
            padding: 120px 0;
            align-items: center;
          }
        }

        .btn-launch {
          font-family: var(--font-mono); font-size: 13px; letter-spacing: 2.5px;
          color: #fff; background: transparent; border: 1px solid #fff;
          border-radius: 9999px; padding: 14px 32px; cursor: pointer;
          text-transform: uppercase; display: inline-flex; align-items: center; gap: 10px;
          transition: background 0.2s, color 0.2s;
        }
        .btn-launch:hover { background: #fff; color: #000; }

        .btn-nav {
          font-family: var(--font-mono); font-size: 11px; letter-spacing: 2.5px;
          color: #fff; background: transparent; border: 1px solid #333;
          border-radius: 9999px; padding: 8px 20px; cursor: pointer;
          text-transform: uppercase; transition: border-color 0.2s;
        }
        .btn-nav:hover { border-color: #fff; }
      `}</style>

      {/* ── Nav ──────────────────────────────────────────────────── */}
      <nav className="land-nav">
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 13, letterSpacing: '6px', color: '#fff', textTransform: 'uppercase' }}>
          ResumeAI
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#555', textTransform: 'uppercase', display: 'none' }}
            className="hidden-mobile">
            Career Advisor
          </span>
          <button onClick={onStart} className="btn-nav">Launch</button>
        </div>
      </nav>

      {/* ── Hero ─────────────────────────────────────────────────── */}
      <section>
        <div className="land-section">
          <div className="hero-grid">
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#555', textTransform: 'uppercase', marginBottom: 28 }}>
                AI-Powered Career Intelligence
              </p>
              <h1 style={{
                fontFamily: 'var(--font-display)', fontWeight: 400,
                fontSize: 'clamp(40px, 7vw, 80px)', letterSpacing: '3px',
                textTransform: 'uppercase', color: '#fff',
                lineHeight: 1.05, margin: 0,
              }}>
                Know Exactly<br />
                What's Missing<br />
                <span style={{ color: '#c3d9f3' }}>In Your Career.</span>
              </h1>
            </div>

            <div>
              <div style={{ width: 1, height: 32, background: '#262626', marginBottom: 28 }} />
              <p style={{ fontFamily: 'var(--font-body)', fontSize: 16, lineHeight: 1.75, color: '#888', marginBottom: 36, letterSpacing: 0 }}>
                Upload your resume. The AI agent scans live job postings across every major platform
                and tells you precisely which skills are standing between you and your target role.
              </p>
              <button onClick={onStart} className="btn-launch">
                Analyze My Resume <ArrowRight size={14} />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats ────────────────────────────────────────────────── */}
      <div style={{ borderTop: '1px solid #1a1a1a', background: '#080808' }}>
        <div className="land-section">
          <div className="stats-grid">
            {STATS.map((s, i) => (
              <div key={i} className="stat-cell">
                <div style={{ fontFamily: 'var(--font-display)', fontSize: 28, letterSpacing: '2px', color: '#fff', textTransform: 'uppercase' }}>
                  {s.value}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#444', textTransform: 'uppercase', marginTop: 6 }}>
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── How it works ─────────────────────────────────────────── */}
      <section style={{ borderTop: '1px solid #1a1a1a', padding: '80px 0' }}>
        <div className="land-section">
          <div className="section-header">
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#444', textTransform: 'uppercase', marginBottom: 14 }}>Process</p>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 32, letterSpacing: '2px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, margin: 0, lineHeight: 1.2 }}>
                How It Works
              </h2>
            </div>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.75, color: '#666', letterSpacing: 0, margin: 0 }}>
              Two AI agents work in sequence — one to understand your profile, one to scan the live market —
              and together produce a complete picture of the gap between where you are and where you want to be.
            </p>
          </div>

          <div style={{ borderTop: '1px solid #1a1a1a' }}>
            {FEATURES.map((f, i) => (
              <div key={i} className="feature-row">
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#2a2a2a', textTransform: 'uppercase', paddingTop: 2 }}>
                  {f.number}
                </span>
                <div>
                  <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#c3d9f3', textTransform: 'uppercase', marginBottom: 8 }}>
                    {f.label}
                  </p>
                  <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 18, letterSpacing: '1.5px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, margin: '0 0 12px' }}>
                    {f.title}
                  </h3>
                  <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.7, color: '#666', letterSpacing: 0, margin: 0 }}
                    className="feature-body-mobile">
                    {f.body}
                  </p>
                </div>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.7, color: '#666', letterSpacing: 0, margin: 0, display: 'none' }}
                  className="feature-body-desktop">
                  {f.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Agents ───────────────────────────────────────────────── */}
      <section style={{ borderTop: '1px solid #1a1a1a', background: '#050505', padding: '80px 0' }}>
        <div className="land-section">
          <div className="section-header">
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#444', textTransform: 'uppercase', marginBottom: 14 }}>Agents</p>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 32, letterSpacing: '2px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, margin: 0, lineHeight: 1.2 }}>
                Built for Precision
              </h2>
            </div>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.75, color: '#666', letterSpacing: 0, margin: 0 }}>
              Each agent has a single focused responsibility — no shared context, no compromise.
            </p>
          </div>

          <div className="agents-grid">
            {AGENTS.map((agent, i) => (
              <div key={i} style={{ background: '#000', padding: '40px 32px' }}>
                <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#c3d9f3', textTransform: 'uppercase', marginBottom: 14 }}>
                  {agent.tag}
                </p>
                <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 22, letterSpacing: '2px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, marginBottom: 16 }}>
                  {agent.title}
                </h3>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.7, color: '#666', letterSpacing: 0, marginBottom: 28 }}>
                  {agent.desc}
                </p>
                <div style={{ borderTop: '1px solid #141414', paddingTop: 20 }}>
                  {agent.items.map(item => (
                    <div key={item} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '9px 0', borderBottom: '1px solid #0d0d0d' }}>
                      <span style={{ width: 3, height: 3, background: '#c3d9f3', flexShrink: 0 }} />
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#444', textTransform: 'uppercase' }}>
                        {item}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────── */}
      <section style={{ borderTop: '1px solid #1a1a1a' }}>
        <div className="land-section">
          <div className="cta-grid">
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#444', textTransform: 'uppercase', marginBottom: 20 }}>
                Get Started
              </p>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(32px, 4vw, 52px)', letterSpacing: '3px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, margin: 0, lineHeight: 1.1 }}>
                Your Next Role<br />Starts Here.
              </h2>
            </div>
            <div>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.75, color: '#666', letterSpacing: 0, marginBottom: 36 }}>
                Upload your resume. Enter your target role. The agents do the rest — scanning live job postings,
                identifying your gaps, and mapping the fastest route to your next opportunity.
              </p>
              <button onClick={onStart} className="btn-launch">
                Begin Analysis <ArrowRight size={14} />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <footer style={{ borderTop: '1px solid #1a1a1a', padding: '32px 24px' }}>
        <div className="land-section" style={{ padding: 0 }}>
          <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
              <span style={{ fontFamily: 'var(--font-display)', fontSize: 12, letterSpacing: '6px', color: '#2a2a2a', textTransform: 'uppercase' }}>
                ResumeAI
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#2a2a2a', textTransform: 'uppercase' }}>
                2026
              </span>
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#2a2a2a', textTransform: 'uppercase' }}>
              Powered by Google Gemini · JSearch · LangGraph
            </span>
          </div>
        </div>
      </footer>
    </div>
  )
}
