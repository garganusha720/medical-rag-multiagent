'use client'

import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  Plus,
  MessageSquare,
  LogOut,
  HeartPulse,
} from 'lucide-react'
import { createClient } from '@/lib/supabase/client'
import { useRouter } from 'next/navigation'

export type Session = {
  id: string
  title: string
  created_at: string
}

type Props = {
  sessions: Session[]
  activeSessionId: string | null
  onSelectSession: (id: string) => void
  onNewChat: () => void
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
}: Props) {
  const router = useRouter()
  const supabase = createClient()

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.push('/login')
  }

  return (
    <aside className="flex h-screen w-[280px] shrink-0 flex-col border-r border-[#dce4f4] bg-[#eef3fc]">

      {/* Logo */}
      <div className="px-5 pt-6 pb-5">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-[#5937f5] to-[#7548ff] shadow-lg shadow-purple-200">
            <HeartPulse
              className="text-white"
              size={24}
              strokeWidth={2.2}
            />
          </div>

          <div>
            <h1 className="text-[22px] font-bold leading-none text-[#101c42]">
              MedRAG
            </h1>

            <p className="mt-1 text-xs font-medium text-[#69799e]">
              Medical AI Assistant
            </p>
          </div>
        </div>
      </div>

      {/* New chat */}
      <div className="px-4">
        <Button
          onClick={onNewChat}
          className="h-11 w-full justify-start gap-3 rounded-xl bg-[#5937f5] text-white shadow-md shadow-purple-200 transition hover:bg-[#4d2fe0]"
        >
          <Plus className="h-5 w-5" />
          <span className="font-semibold">New chat</span>
        </Button>
      </div>

      <Separator className="my-5 bg-[#dce4f4]" />

      {/* Chat history */}
      <div className="px-4">
        <p className="mb-3 px-2 text-xs font-semibold uppercase tracking-wider text-[#8290ad]">
          Recent conversations
        </p>
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto px-3">

        {sessions.length === 0 && (
          <div className="px-3 py-5">
            <div className="rounded-xl border border-dashed border-[#ccd6ea] bg-white/50 p-4 text-center">
              <MessageSquare className="mx-auto mb-2 h-5 w-5 text-[#9aa8c3]" />

              <p className="text-sm font-medium text-[#657292]">
                No chats yet
              </p>

              <p className="mt-1 text-xs text-[#94a0b8]">
                Start a new medical conversation
              </p>
            </div>
          </div>
        )}

        {sessions.map((session) => (
          <button
            key={session.id}
            onClick={() => onSelectSession(session.id)}
            className={`group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition ${
              activeSessionId === session.id
                ? 'bg-white text-[#5033dd] shadow-sm'
                : 'text-[#536384] hover:bg-white/70'
            }`}
          >
            <MessageSquare
              className={`h-4 w-4 shrink-0 ${
                activeSessionId === session.id
                  ? 'text-[#5c35f5]'
                  : 'text-[#8996b0]'
              }`}
            />

            <span className="truncate font-medium">
              {session.title}
            </span>
          </button>
        ))}
      </div>

      {/* Bottom */}
      <div className="border-t border-[#dce4f4] p-4">
        <Button
          variant="ghost"
          onClick={handleLogout}
          className="h-11 w-full justify-start gap-3 rounded-xl text-[#687795] hover:bg-white hover:text-[#d33d55]"
        >
          <LogOut className="h-4 w-4" />
          Log out
        </Button>
      </div>
    </aside>
  )
}