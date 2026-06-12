import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { X } from 'lucide-react'

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
      <div className="flex items-center justify-between py-3 hairline-bottom">
        <span className="font-body text-ink text-sm truncate">{file.name}</span>
        <button
          onClick={() => onFile(null)}
          className="ml-4 font-mono text-xs text-muted hover:text-ink transition-colors uppercase tracking-caption flex items-center gap-1"
        >
          <X size={12} /> Remove
        </button>
      </div>
    )
  }

  return (
    <div
      {...getRootProps()}
      className="py-4 cursor-pointer transition-colors"
      style={{
        borderBottom: `1px solid ${isDragActive ? '#ffffff' : '#3a3a3a'}`,
      }}
    >
      <input {...getInputProps()} />
      <p className="font-body text-sm" style={{ color: isDragActive ? '#ffffff' : '#666666', letterSpacing: 0 }}>
        {isDragActive ? 'Drop your resume here' : 'Drag & drop PDF, or click to browse'}
      </p>
    </div>
  )
}
