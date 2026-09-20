'use client'

import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react'

export type Citation = {
  marker: string
  chunk_id: string
  source: string
  text: string
  url: string
  score: number
}

export function CitationCard({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="rounded-xl border border-[#e1e6f0] bg-[#f8faff] text-sm">

      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between px-3 py-2.5 text-left transition hover:bg-white"
      >

        <div className="flex items-center gap-2">

          <Badge
            variant="secondary"
            className="bg-[#ebe7ff] text-[#5c35f5]"
          >
            [{citation.marker}]
          </Badge>

          <span className="font-medium capitalize text-[#697795]">
            {citation.source}
          </span>

        </div>

        {expanded ? (
          <ChevronUp className="h-4 w-4 text-[#7e8aa5]" />
        ) : (
          <ChevronDown className="h-4 w-4 text-[#7e8aa5]" />
        )}

      </button>

      {expanded && (
        <div className="border-t border-[#e5e9f1] px-3 pb-3 pt-3">

          <p className="text-[13px] leading-5 text-[#687692]">
            {citation.text}
          </p>

          <a
            href={citation.url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 flex items-center gap-1 text-xs font-semibold text-[#5937f5] hover:underline"
          >
            View source
            <ExternalLink className="h-3 w-3" />
          </a>

        </div>
      )}

    </div>
  )
}