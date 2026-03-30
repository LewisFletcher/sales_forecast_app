import { useState, useRef, useEffect } from 'react'

const AGENT_URL = 'http://localhost:8001'

const INITIAL_MESSAGE = {
  role: 'assistant',
  content: "Hello! I'm your sales forecast assistant. Ask me for a sales forecast — you can specify a date, country, product category, and device type for the most accurate prediction.",
}

export default function App() {
  const [messages, setMessages] = useState([INITIAL_MESSAGE])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const threadId = useRef(crypto.randomUUID())
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const res = await fetch(`${AGENT_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, thread_id: threadId.current }),
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`)
      }

      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}. Make sure both the API (port 8000) and agent (port 8001) servers are running.` },
      ])
    } finally {
      setLoading(false)
      textareaRef.current?.focus()
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleInput = (e) => {
    setInput(e.target.value)
    // Auto-resize textarea
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  return (
    <div className="layout">
      <header className="header">
        <div className="header-inner">
          <span className="header-icon">📊</span>
          <h1>Sales Forecast Assistant</h1>
        </div>
      </header>

      <main className="messages-container">
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message-row ${msg.role}`}>
              <div className="bubble">
                {msg.content.split('\n').map((line, j) => (
                  <span key={j}>
                    {line}
                    {j < msg.content.split('\n').length - 1 && <br />}
                  </span>
                ))}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-row assistant">
              <div className="bubble typing">
                <span /><span /><span />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="input-bar">
        <div className="input-inner">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask for a sales forecast… (Enter to send, Shift+Enter for new line)"
            disabled={loading}
            rows={1}
            className="input-textarea"
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="send-btn"
          >
            Send
          </button>
        </div>
      </footer>
    </div>
  )
}
