import { MapPin, Clock, Wifi, ExternalLink, Building2 } from 'lucide-react'

function formatDate(dateStr) {
  if (!dateStr || dateStr === 'N/A') return 'N/A'
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric'
    })
  } catch {
    return dateStr.slice(0, 10)
  }
}

const TYPE_COLORS = {
  FULLTIME:   'bg-green-900/60 text-green-300 border-green-700',
  INTERN:     'bg-purple-900/60 text-purple-300 border-purple-700',
  PARTTIME:   'bg-yellow-900/60 text-yellow-300 border-yellow-700',
  CONTRACTOR: 'bg-orange-900/60 text-orange-300 border-orange-700',
}

export default function JobCard({ job }) {
  const typeColor = TYPE_COLORS[job.employment_type] || 'bg-slate-800 text-slate-300 border-slate-600'

  return (
    <div className="group rounded-2xl bg-slate-900 border border-slate-800
                    hover:border-brand-700 transition-all duration-200
                    p-5 flex flex-col gap-4">

      {/* Top row */}
      <div className="flex items-start gap-4">
        {/* Logo */}
        <div className="w-12 h-12 rounded-xl bg-slate-800 border border-slate-700
                        flex items-center justify-center shrink-0 overflow-hidden">
          {job.logo
            ? <img src={job.logo} alt={job.company} className="w-10 h-10 object-contain" />
            : <Building2 size={22} className="text-slate-500" />
          }
        </div>

        {/* Title + company */}
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-slate-100 text-base leading-snug truncate">
            {job.title}
          </h3>
          <p className="text-slate-400 text-sm mt-0.5 truncate">{job.company}</p>
        </div>

        {/* Type badge */}
        <span className={`shrink-0 text-xs font-semibold px-2.5 py-1
                          rounded-full border ${typeColor}`}>
          {job.employment_type}
        </span>
      </div>

      {/* Meta row */}
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <MapPin size={12} /> {job.location}
        </span>
        {job.is_remote && (
          <span className="flex items-center gap-1 text-sky-400">
            <Wifi size={12} /> Remote
          </span>
        )}
        <span className="flex items-center gap-1">
          <Clock size={12} /> {formatDate(job.posted)}
        </span>
      </div>

      {/* Description snippet */}
      {job.description && (
        <p className="text-slate-500 text-xs line-clamp-2 leading-relaxed">
          {job.description}
        </p>
      )}

      {/* Apply button */}
      {job.apply_link
        ? (
          <a
            href={job.apply_link}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-auto flex items-center justify-center gap-2
                       rounded-xl bg-brand-600 hover:bg-brand-500
                       text-white text-sm font-semibold py-2.5
                       transition-colors duration-150"
          >
            Apply Now <ExternalLink size={13} />
          </a>
        ) : (
          <span className="mt-auto text-center text-xs text-slate-600 py-2">
            No apply link available
          </span>
        )
      }
    </div>
  )
}
