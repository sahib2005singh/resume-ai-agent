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

export default function Landing({ onStart }) {
  return (
    <div style={{ background: '#000', minHeight: '100vh', color: '#ccc' }}>

      {/* ── Nav ──────────────────────────────────────────────────── */}
      <nav style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50,
        height: 56, display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', padding: '0 48px',
        borderBottom: '1px solid #1a1a1a',
        background: 'rgba(0,0,0,0.92)', backdropFilter: 'blur(12px)',
      }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: 13, letterSpacing: '6px', color: '#fff', textTransform: 'uppercase' }}>
          ResumeAI
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 32 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#666', textTransform: 'uppercase' }}>
            Career Advisor
          </span>
          <button onClick={onStart} style={{
            fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2.5px',
            color: '#fff', background: 'transparent', border: '1px solid #333',
            borderRadius: 9999, padding: '8px 20px', cursor: 'pointer',
            textTransform: 'uppercase', transition: 'border-color 0.2s',
          }}
            onMouseEnter={e => e.target.style.borderColor = '#fff'}
            onMouseLeave={e => e.target.style.borderColor = '#333'}
          >
            Launch
          </button>
        </div>
      </nav>

      {/* ── Hero ─────────────────────────────────────────────────── */}
      <section style={{ paddingTop: 160, paddingBottom: 120, paddingLeft: 48, paddingRight: 48, maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 80, alignItems: 'end' }}>

          {/* Left — headline */}
          <div>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#666', textTransform: 'uppercase', marginBottom: 32 }}>
              AI-Powered Career Intelligence
            </p>
            <h1 style={{
              fontFamily: 'var(--font-display)', fontWeight: 400,
              fontSize: 'clamp(48px, 6vw, 80px)', letterSpacing: '3px',
              textTransform: 'uppercase', color: '#fff',
              lineHeight: 1.05, margin: 0,
            }}>
              Know Exactly<br />
              What's Missing<br />
              <span style={{ color: '#c3d9f3' }}>In Your Career.</span>
            </h1>
          </div>

          {/* Right — description + CTA */}
          <div style={{ paddingBottom: 8 }}>
            <div style={{ width: 1, height: 48, background: '#262626', marginBottom: 32 }} />
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 16, lineHeight: 1.7, color: '#999', marginBottom: 40, letterSpacing: 0 }}>
              Upload your resume. The AI agent scans live job postings across every major platform
              and tells you precisely which skills are standing between you and your target role.
            </p>
            <button onClick={onStart} style={{
              fontFamily: 'var(--font-mono)', fontSize: 13, letterSpacing: '2.5px',
              color: '#fff', background: 'transparent', border: '1px solid #fff',
              borderRadius: 9999, padding: '14px 32px', cursor: 'pointer',
              textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 10,
              transition: 'background 0.2s, color 0.2s',
            }}
              onMouseEnter={e => { e.currentTarget.style.background = '#fff'; e.currentTarget.style.color = '#000' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#fff' }}
            >
              Analyze My Resume <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </section>

      {/* ── Divider with stats ───────────────────────────────────── */}
      <div style={{ borderTop: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a', background: '#0a0a0a' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 48px', display: 'flex' }}>
          {[
            { value: '10+',    label: 'Job Platforms' },
            { value: 'Live',   label: 'Real-Time Data' },
            { value: 'Gemini', label: 'AI Engine' },
            { value: '2',      label: 'Dedicated Agents' },
          ].map((s, i) => (
            <div key={i} style={{
              flex: 1, padding: '36px 0',
              borderRight: i < 3 ? '1px solid #1a1a1a' : 'none',
              paddingLeft: i === 0 ? 0 : 40,
            }}>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: 28, letterSpacing: '2px', color: '#fff', textTransform: 'uppercase' }}>
                {s.value}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#555', textTransform: 'uppercase', marginTop: 6 }}>
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── How it works ─────────────────────────────────────────── */}
      <section style={{ maxWidth: 1200, margin: '0 auto', padding: '120px 48px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 80, marginBottom: 80 }}>
          <div>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#555', textTransform: 'uppercase', marginBottom: 16 }}>
              Process
            </p>
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 32, letterSpacing: '2px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, margin: 0, lineHeight: 1.2 }}>
              How It<br />Works
            </h2>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-end' }}>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.7, color: '#777', maxWidth: 480, letterSpacing: 0, margin: 0 }}>
              Two AI agents work in sequence — one to understand your profile, one to scan the live market —
              and together produce a complete picture of the gap between where you are and where you want to be.
            </p>
          </div>
        </div>

        <div style={{ borderTop: '1px solid #1a1a1a' }}>
          {FEATURES.map((f, i) => (
            <div key={i} style={{
              display: 'grid', gridTemplateColumns: '80px 280px 1fr',
              gap: 48, padding: '40px 0',
              borderBottom: '1px solid #1a1a1a',
              alignItems: 'start',
            }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#333', textTransform: 'uppercase', paddingTop: 4 }}>
                {f.number}
              </span>
              <div>
                <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#c3d9f3', textTransform: 'uppercase', marginBottom: 8 }}>
                  {f.label}
                </p>
                <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 18, letterSpacing: '1.5px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, margin: 0 }}>
                  {f.title}
                </h3>
              </div>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.7, color: '#777', letterSpacing: 0, margin: 0, maxWidth: 520 }}>
                {f.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Two agents ───────────────────────────────────────────── */}
      <section style={{ borderTop: '1px solid #1a1a1a', background: '#050505' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', padding: '120px 48px' }}>

          <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 80, marginBottom: 80 }}>
            <div>
              <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#555', textTransform: 'uppercase', marginBottom: 16 }}>
                Agents
              </p>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 32, letterSpacing: '2px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, margin: 0, lineHeight: 1.2 }}>
                Built for<br />Precision
              </h2>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.7, color: '#777', letterSpacing: 0, margin: 0, maxWidth: 480 }}>
                Each agent has a single focused responsibility — no shared context, no compromise.
              </p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: '#1a1a1a' }}>
            {[
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
            ].map((agent, i) => (
              <div key={i} style={{ background: '#000', padding: '48px' }}>
                <p style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#c3d9f3', textTransform: 'uppercase', marginBottom: 16 }}>
                  {agent.tag}
                </p>
                <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 24, letterSpacing: '2px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, marginBottom: 20 }}>
                  {agent.title}
                </h3>
                <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.7, color: '#777', letterSpacing: 0, marginBottom: 32 }}>
                  {agent.desc}
                </p>
                <div style={{ borderTop: '1px solid #1a1a1a', paddingTop: 24 }}>
                  {agent.items.map(item => (
                    <div key={item} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 0', borderBottom: '1px solid #0d0d0d' }}>
                      <span style={{ width: 3, height: 3, background: '#c3d9f3', flexShrink: 0 }} />
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#555', textTransform: 'uppercase' }}>
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
        <div style={{ maxWidth: 1200, margin: '0 auto', padding: '120px 48px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80, alignItems: 'center' }}>
          <div>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '2px', color: '#555', textTransform: 'uppercase', marginBottom: 20 }}>
              Get Started
            </p>
            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(36px, 4vw, 56px)', letterSpacing: '3px', color: '#fff', textTransform: 'uppercase', fontWeight: 400, margin: 0, lineHeight: 1.1 }}>
              Your Next Role<br />Starts Here.
            </h2>
          </div>
          <div>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 15, lineHeight: 1.7, color: '#777', letterSpacing: 0, marginBottom: 40 }}>
              Upload your resume. Enter your target role. The agents do the rest — scanning live job postings,
              identifying your gaps, and mapping the fastest route to your next opportunity.
            </p>
            <button onClick={onStart} style={{
              fontFamily: 'var(--font-mono)', fontSize: 13, letterSpacing: '2.5px',
              color: '#fff', background: 'transparent', border: '1px solid #fff',
              borderRadius: 9999, padding: '14px 32px', cursor: 'pointer',
              textTransform: 'uppercase', display: 'inline-flex', alignItems: 'center', gap: 10,
              transition: 'background 0.2s, color 0.2s',
            }}
              onMouseEnter={e => { e.currentTarget.style.background = '#fff'; e.currentTarget.style.color = '#000' }}
              onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#fff' }}
            >
              Begin Analysis <ArrowRight size={14} />
            </button>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <footer style={{ borderTop: '1px solid #1a1a1a', padding: '40px 48px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: 12, letterSpacing: '6px', color: '#333', textTransform: 'uppercase' }}>
            ResumeAI
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#333', textTransform: 'uppercase' }}>
            Powered by Google Gemini · JSearch · LangGraph
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '2px', color: '#333', textTransform: 'uppercase' }}>
            2026
          </span>
        </div>
      </footer>
    </div>
  )
}
