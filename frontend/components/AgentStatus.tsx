'use client'

import { Search, Filter, Sparkles, CheckCircle2 } from 'lucide-react'

export type AgentStage = 'retrieving' | 'critiquing' | 'generating' | 'done'

const stages: { key: AgentStage; label: string; icon: React.ElementType }[] = [
  { key: 'retrieving', label: 'Retrieving sources...', icon: Search },
  { key: 'critiquing', label: 'Critiquing relevance...', icon: Filter },
  { key: 'generating', label: 'Generating answer...', icon: Sparkles },
]

export function AgentStatus({ stage }: { stage: AgentStage }) {
  if (stage === 'done') return null

  const current = stages.find((s) => s.key === stage) ?? stages[0]
  const Icon = current.icon

  return (
    <div className="flex items-center gap-2 rounded-lg bg-gray-100 p-3 text-sm text-muted-foreground">
      <Icon className="h-4 w-4 animate-pulse" />
      <span>{current.label}</span>
    </div>
  )
}