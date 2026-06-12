import { MapPin, ExternalLink, Building2 } from 'lucide-react'

function formatDate(dateStr) {
  if (!dateStr || dateStr === 'N/A') return null
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric'
    })
  } catch { return null }
}

export default function JobCard({ job }) {
  const date = formatDate(job.posted)

  return (
    <div
      className="group flex flex-col"
      style={{
        background: '#000000',
        border: '1px solid #262626',
        padding: 0,
        transition: 'border-color 0.2s',
      }}
      onMouseEnter={e => e.currentTarget.style.borderColor = '#3a3a3a'}
      onMouseLeave={e => e.currentTarget.style.borderColor = '#262626'}
    >
      {/* Top section */}
      <div style={{ padding: '24px', borderBottom: '1px solid #141414' }}>
        {/* Logo + company */}
        <div className="flex items-start gap-3 mb-4">
          <div className="flex items-center justify-center shrink-0"
            style={{ width: 36, height: 36, background: '#141414', border: '1px solid #262626' }}>
            {job.logo
              ? <img src={job.logo} alt={job.company} style={{ width: 28, height: 28, objectFit: 'contain' }} />
              : <Building2 size={16} color="#666666" />
            }
          </div>
          <div className="min-w-0">
            <p className="font-mono text-xs uppercase tracking-caption text-muted truncate">{job.company}</p>
          </div>
        </div>

        {/* Title */}
        <h3 className="font-display text-ink uppercase mb-3"
          style={{ fontSize: 16, letterSpacing: '1.5px', fontWeight: 400, lineHeight: 1.3 }}>
          {job.title}
        </h3>

        {/* Meta */}
        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
          <span className="flex items-center gap-1.5 caption-label" style={{ fontSize: 10 }}>
            <MapPin size={10} /> {job.location}
          </span>
          {job.is_remote && (
            <span className="caption-label" style={{ fontSize: 10, color: '#c3d9f3' }}>Remote</span>
          )}
          <span className="caption-label" style={{ fontSize: 10 }}>{job.employment_type}</span>
        </div>
      </div>

      {/* Description */}
      {job.description && (
        <div style={{ padding: '16px 24px', borderBottom: '1px solid #141414', flex: 1 }}>
          <p className="font-body text-muted" style={{ fontSize: 13, lineHeight: 1.6, letterSpacing: 0 }}>
            {job.description.slice(0, 200)}{job.description.length > 200 ? '…' : ''}
          </p>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between" style={{ padding: '16px 24px' }}>
        {date
          ? <span className="caption-label" style={{ fontSize: 10, color: '#444444' }}>{date}</span>
          : <span />
        }
        {job.apply_link
          ? (
            <a
              href={job.apply_link}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-primary"
              style={{ height: 36, padding: '0 20px', fontSize: 11, letterSpacing: '2px' }}
            >
              Apply <ExternalLink size={11} />
            </a>
          ) : (
            <span className="caption-label" style={{ color: '#333333', fontSize: 10 }}>No link</span>
          )
        }
      </div>
    </div>
  )
}
