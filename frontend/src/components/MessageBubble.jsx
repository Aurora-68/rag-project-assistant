import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";

function CodeBlock({ className, children, ...props }) {
  const match = /language-(\w+)/.exec(className || "");
  const lang = match ? match[1] : null;
  const code = String(children).replace(/\n$/, "");

  if (lang) {
    return (
      <SyntaxHighlighter
        style={oneLight}
        language={lang}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: "8px",
          fontSize: "0.82em",
        }}
      >
        {code}
      </SyntaxHighlighter>
    );
  }

  return (
    <code className={className} {...props}>
      {children}
    </code>
  );
}

function SourceTag({ source }) {
  const name = source.source_file?.split("/").pop() || "Unknown";
  const page = source.page_number != null ? ` · p.${source.page_number}` : "";
  return (
    <span className="source-tag">
      📄 {name}{page}
    </span>
  );
}

function ThinkingDots() {
  return (
    <div className="thinking-dots">
      <span /><span /><span />
    </div>
  );
}

export default function MessageBubble({ message }) {
  const { role, content, sources, isStreaming, isError, fileName } = message;
  const isUser = role === "user";
  
  // loosen pill radius for multiline markdown content
  const hasMarkdown = !isUser && content && content.includes("\n");

  return (
    <div className={`message-row ${role}`}>
      {!isUser && (
        <div className="msg-avatar" aria-hidden="true">🎓</div>
      )}

      <div className={`msg-bubble ${role}${isError ? " error" : ""}${hasMarkdown ? " has-markdown" : ""}`}>
        {isUser ? (
          <>
            {fileName && (
              <div style={{ fontSize: "0.75rem", opacity: 0.8, marginBottom: 4 }}>
                📎 {fileName}
              </div>
            )}
            <span>{content}</span>
          </>
        ) : content ? (
          <div className="md-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code: ({ node, inline, className, children, ...props }) => {
                  if (inline) {
                    return <code className={className} {...props}>{children}</code>;
                  }
                  return <CodeBlock className={className} {...props}>{children}</CodeBlock>;
                },
              }}
            >
              {content}
            </ReactMarkdown>
            {isStreaming && <span className="typing-cursor" />}
          </div>
        ) : isStreaming ? (
          <ThinkingDots />
        ) : null}

        {!isUser && !isStreaming && sources?.length > 0 && (
          <div className="source-chips">
            {sources.map((s, i) => (
              <SourceTag key={i} source={s} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
