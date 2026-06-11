import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { FileText, UploadCloud, X } from 'lucide-react'

export default function ResumeUpload({ file, onFile }) {
  const onDrop = useCallback(accepted => {
    if (accepted[0]) onFile(accepted[0])
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1,
  })

  if (file) {
    return (
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Resume
        </label>
        <div className="h-12 px-4 rounded-xl bg-slate-800 border border-brand-600
                        flex items-center gap-3 text-sm">
          <FileText size={16} className="text-brand-400 shrink-0" />
          <span className="truncate text-slate-200 flex-1">{file.name}</span>
          <button
            onClick={() => onFile(null)}
            className="text-slate-500 hover:text-red-400 transition"
          >
            <X size={15} />
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
        Resume (PDF)
      </label>
      <div
        {...getRootProps()}
        className={`h-12 px-4 rounded-xl border-2 border-dashed cursor-pointer
                    flex items-center gap-2 text-sm transition
                    ${isDragActive
                      ? 'border-brand-400 bg-brand-950 text-brand-300'
                      : 'border-slate-700 bg-slate-800 text-slate-500 hover:border-slate-500 hover:text-slate-300'}`}
      >
        <input {...getInputProps()} />
        <UploadCloud size={16} className="shrink-0" />
        <span>{isDragActive ? 'Drop it here' : 'Drag & drop or click to upload'}</span>
      </div>
    </div>
  )
}
