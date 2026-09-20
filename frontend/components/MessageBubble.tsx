'use client'

import { CitationCard, type Citation } from './CitationCard'
import { StreamingDots } from './StreamingDots'

type Message = {
  role: string
  content: string
  citations?: Citation[]
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  const isEmpty = !isUser && message.content === ''
  const parts = message.content.split(/(\[c\d+\])/g)

  return (
    <div className={`p-3 rounded-lg max-w-md ${isUser ? 'bg-blue-100 ml-auto' : 'bg-gray-100'}`}>
      {isEmpty ? (
        <StreamingDots />
      ) : (
        <p>
          {parts.map((part, i) => {
            const match = part.match(/\[c(\d+)\]/)
            if (match) {
              return (
                <sup key={i} className="text-blue-600 font-semibold mx-0.5">
                  {part}
                </sup>
              )
            }
            return <span key={i}>{part}</span>
          })}
        </p>
      )}

      {!isUser && message.citations && message.citations.length > 0 && (
        <div className="mt-3 space-y-1">
          {message.citations.map((c) => (
            <CitationCard key={c.marker} citation={c} />
          ))}
        </div>
      )}
    </div>
  )
}