import ReactMarkdown from 'react-markdown'

export default function AnalysisResult({ content }) {
  return (
    <div className="prose prose-invert prose-sm max-w-none
                    prose-headings:text-slate-100 prose-headings:font-bold
                    prose-h3:text-xl prose-h4:text-base
                    prose-p:text-slate-300 prose-p:leading-relaxed
                    prose-li:text-slate-300
                    prose-strong:text-slate-100
                    prose-table:text-sm
                    prose-th:text-slate-200 prose-th:bg-slate-800 prose-th:px-3 prose-th:py-2
                    prose-td:text-slate-300 prose-td:px-3 prose-td:py-2
                    prose-hr:border-slate-700
                    [&_table]:w-full [&_table]:border-collapse
                    [&_table]:rounded-xl [&_table]:overflow-hidden
                    [&_th]:border [&_th]:border-slate-700
                    [&_td]:border [&_td]:border-slate-800
                    [&_tr:nth-child(even)_td]:bg-slate-800/50">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  )
}
