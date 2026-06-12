import ReactMarkdown from 'react-markdown'

export default function AnalysisResult({ content }) {
  return (
    <div className="font-body text-body leading-relaxed space-y-4"
      style={{
        '--tw-prose-body': '#cccccc',
        fontSize: 15,
        letterSpacing: 0,
      }}
    >
      <style>{`
        .analysis-content h3 {
          font-family: var(--font-display);
          font-size: 20px;
          letter-spacing: 1px;
          text-transform: uppercase;
          color: #ffffff;
          font-weight: 400;
          margin-top: 32px;
          margin-bottom: 12px;
        }
        .analysis-content h4 {
          font-family: var(--font-display);
          font-size: 14px;
          letter-spacing: 2px;
          text-transform: uppercase;
          color: #999999;
          font-weight: 400;
          margin-top: 24px;
          margin-bottom: 8px;
        }
        .analysis-content p {
          color: #cccccc;
          font-family: var(--font-body);
          font-size: 15px;
          line-height: 1.7;
          margin-bottom: 12px;
          letter-spacing: 0;
        }
        .analysis-content strong {
          color: #ffffff;
          font-weight: 400;
        }
        .analysis-content ul, .analysis-content ol {
          margin: 12px 0;
          padding-left: 0;
          list-style: none;
        }
        .analysis-content li {
          color: #cccccc;
          font-size: 15px;
          padding: 6px 0;
          border-bottom: 1px solid #141414;
          padding-left: 16px;
          position: relative;
        }
        .analysis-content li::before {
          content: '→';
          position: absolute;
          left: 0;
          color: #c3d9f3;
          font-family: var(--font-mono);
          font-size: 12px;
        }
        .analysis-content table {
          width: 100%;
          border-collapse: collapse;
          margin: 24px 0;
          font-size: 13px;
        }
        .analysis-content th {
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 2px;
          text-transform: uppercase;
          color: #999999;
          text-align: left;
          padding: 12px 16px;
          border-bottom: 1px solid #262626;
          font-weight: 400;
        }
        .analysis-content td {
          color: #cccccc;
          font-family: var(--font-body);
          padding: 12px 16px;
          border-bottom: 1px solid #141414;
          font-size: 14px;
        }
        .analysis-content tr:hover td {
          background: #0d0d0d;
        }
        .analysis-content hr {
          border: none;
          border-top: 1px solid #262626;
          margin: 32px 0;
        }
        .analysis-content code {
          font-family: var(--font-mono);
          font-size: 12px;
          color: #c3d9f3;
          background: #141414;
          padding: 2px 8px;
        }
      `}</style>
      <div className="analysis-content">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </div>
  )
}
