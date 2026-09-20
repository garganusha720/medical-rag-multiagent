'use client'

import { useState } from 'react'
import { SessionSidebar, type Session } from '@/components/SessionSidebar'
import { ChatWindow } from '@/components/ChatWindow'

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)

  const handleNewChat = () => {
    setActiveSessionId(null)
  }

  const handleSelectSession = (id: string) => {
    setActiveSessionId(id)
  }

  const handleSessionCreated = (session: Session) => {
    setSessions((prev) => [session, ...prev])
    setActiveSessionId(session.id)
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#f4f7ff]">
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />

      <ChatWindow
        activeSessionId={activeSessionId}
        onSessionCreated={handleSessionCreated}
      />
    </div>
  )
}