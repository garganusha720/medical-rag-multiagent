'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { MessageBubble } from '@/components/MessageBubble'
import { type Citation } from '@/components/CitationCard'
import { AgentStatus, type AgentStage } from '@/components/AgentStatus'
import { AlertCircle, HeartPulse } from 'lucide-react'
import { type Session } from '@/components/SessionSidebar'

type Message = {
  role: string
  content: string
  citations?: Citation[]
}

const DRAFT_KEY = '__draft__'

type Props = {
  activeSessionId: string | null
  onSessionCreated: (session: Session) => void
}

export function ChatWindow({
  activeSessionId,
  onSessionCreated,
}: Props) {
  const [messagesBySession, setMessagesBySession] =
    useState<Record<string, Message[]>>({})

  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [agentStage, setAgentStage] =
    useState<AgentStage>('done')
  const [error, setError] = useState<string | null>(null)

  const activeSessionIdRef = useRef<string | null>(activeSessionId)
  activeSessionIdRef.current = activeSessionId

  const activeKey = activeSessionId ?? DRAFT_KEY
  const messages = messagesBySession[activeKey] ?? []

  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    const trimmedInput = input.trim()

    if (!trimmedInput) {
      setError('Please type a question before sending.')
      return
    }

    if (isLoading) return

    setError(null)

    const userMessage: Message = {
      role: 'user',
      content: trimmedInput,
    }

    const submittedForSessionId =
      activeSessionIdRef.current

    const keyAtSubmit =
      submittedForSessionId ?? DRAFT_KEY

    setMessagesBySession((prev) => ({
      ...prev,
      [keyAtSubmit]: [
        ...(prev[keyAtSubmit] ?? []),
        userMessage,
      ],
    }))

    setInput('')
    setIsLoading(true)

    try {
      setAgentStage('retrieving')

      await new Promise((r) => setTimeout(r, 500))

      setAgentStage('critiquing')

      await new Promise((r) => setTimeout(r, 500))

      setAgentStage('generating')

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          session_id: submittedForSessionId,
        }),
      })

      if (!res.ok) {
        throw new Error(
          `Server responded with status ${res.status}`
        )
      }

      const reader = res.body?.getReader()

      if (!reader) {
        throw new Error('No response stream received')
      }

      const decoder = new TextDecoder()

      let assistantText = ''
      let citations: Citation[] = []
      let newSessionId = submittedForSessionId
      let firstChunk = true

      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()

        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Process complete lines from the buffer
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue

          if (firstChunk) {
            setAgentStage('done')

            setMessagesBySession((prev) => ({
              ...prev,
              [keyAtSubmit]: [
                ...(prev[keyAtSubmit] ?? []),
                {
                  role: 'assistant',
                  content: '',
                },
              ],
            }))

            firstChunk = false
          }

          // Vercel AI SDK Text Part: 0:"token text"
          if (line.startsWith('0:')) {
            try {
              const token = JSON.parse(line.slice(2))
              assistantText += token
            } catch {
              assistantText += line.slice(2)
            }
          }

          // Vercel AI SDK Data Part: 2:[{...}]
          if (line.startsWith('2:')) {
            try {
              const dataArray = JSON.parse(line.slice(2))
              const finalData = dataArray[0]
              citations = finalData.citations || []
              newSessionId = finalData.session_id
            } catch {}
          }
        }

        setMessagesBySession((prev) => {
          const current = [
            ...(prev[keyAtSubmit] ?? []),
          ]

          current[current.length - 1] = {
            role: 'assistant',
            content: assistantText,
            citations,
          }

          return {
            ...prev,
            [keyAtSubmit]: current,
          }
        })
      }

      if (!submittedForSessionId && newSessionId) {
        onSessionCreated({
          id: newSessionId,
          title: userMessage.content.slice(0, 40),
          created_at: new Date().toISOString(),
        })

        setMessagesBySession((prev) => {
          const draftMessages =
            prev[DRAFT_KEY] ?? []

          const {
            [DRAFT_KEY]: _,
            ...rest
          } = prev

          return {
            ...rest,
            [newSessionId!]: draftMessages,
          }
        })
      }
    } catch (err) {
      console.error(
        'Chat request failed:',
        err
      )

      setError(
        'Something went wrong while getting a response. Please try again.'
      )

      setMessagesBySession((prev) => {
        const current =
          prev[keyAtSubmit] ?? []

        const trimmed =
          current[
            current.length - 1
          ]?.content === ''
            ? current.slice(0, -1)
            : current

        return {
          ...prev,
          [keyAtSubmit]: trimmed,
        }
      })
    } finally {
      setAgentStage('done')
      setIsLoading(false)
    }
  }

  return (
    <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#f8faff]">

      {/* Subtle background decoration */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -right-32 -top-32 h-[400px] w-[400px] rounded-full bg-[#dfe6ff]/50 blur-3xl" />

        <div className="absolute -bottom-40 left-1/3 h-[350px] w-[350px] rounded-full bg-[#e7e2ff]/40 blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 flex h-[76px] shrink-0 items-center justify-center border-b border-[#e2e7f2] bg-white/80 px-8 backdrop-blur-md">

        <div className="text-center">
          <h1 className="text-[21px] font-bold text-[#101c42]">
            Medical RAG Assistant
          </h1>

          <p className="mt-0.5 text-xs text-[#7b88a5]">
            Evidence-based answers from verified medical sources
          </p>
        </div>

      </header>

      {/* Chat content */}
      <div className="relative z-10 flex min-h-0 flex-1 justify-center">

        <div className="flex w-full max-w-[900px] flex-col px-6">

          {/* Messages */}
          <div className="flex-1 space-y-5 overflow-y-auto py-8">

            {messages.length === 0 && !isLoading && (
              <div className="flex h-full flex-col items-center justify-center text-center">

                {/* Icon */}
                <div className="mb-5 flex h-20 w-20 items-center justify-center rounded-3xl bg-gradient-to-br from-[#ebe7ff] to-[#f3f1ff] shadow-sm">

                  <HeartPulse
                    className="text-[#5c35f5]"
                    size={38}
                    strokeWidth={1.8}
                  />

                </div>

                <h2 className="text-[25px] font-bold text-[#172348]">
                  How can I help you today?
                </h2>

                <p className="mt-2 max-w-[500px] text-[15px] leading-6 text-[#73809d]">
                  Ask a medical question and get answers grounded in
                  verified sources with citations.
                </p>

              </div>
            )}

            {messages.map((m, i) => (
              <MessageBubble
                key={i}
                message={m}
              />
            ))}

            {isLoading &&
              agentStage !== 'done' && (
                <AgentStatus
                  stage={agentStage}
                />
              )}

            <div ref={messagesEndRef} />

          </div>

          {/* Error */}
          {error && (
            <div className="mb-3 flex items-center gap-2 rounded-xl border border-red-100 bg-red-50 p-3 text-sm text-red-700">

              <AlertCircle className="h-4 w-4 shrink-0" />

              <span>{error}</span>

            </div>
          )}

          {/* Input */}
          <div className="pb-6 pt-2">

            <form
              onSubmit={handleSubmit}
              className="flex items-center gap-3 rounded-2xl border border-[#dce3f1] bg-white p-2 shadow-[0_8px_30px_rgba(54,70,120,0.08)]"
            >

              <Input
                value={input}
                onChange={(e) =>
                  setInput(e.target.value)
                }
                placeholder="Ask a medical question..."
                disabled={isLoading}
                className="h-12 border-0 bg-transparent px-4 text-[15px] shadow-none focus-visible:ring-0"
              />

              <Button
                type="submit"
                disabled={isLoading}
                className="h-11 rounded-xl bg-[#5937f5] px-6 font-semibold text-white shadow-md shadow-purple-200 hover:bg-[#4d2fe0]"
              >
                Send
              </Button>

            </form>

            <p className="mt-2 text-center text-[11px] text-[#98a3ba]">
              MedRAG provides information from verified medical sources.
              Always consult a qualified healthcare professional for medical advice.
            </p>

          </div>

        </div>
      </div>

    </main>
  )
}