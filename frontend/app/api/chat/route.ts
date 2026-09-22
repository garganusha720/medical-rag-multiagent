import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function POST(req: NextRequest) {
  const body = await req.json()

  const token = req.headers.get('authorization')

  const backendRes = await fetch(`${BACKEND_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: token } : {}),
    },
    body: JSON.stringify(body),
  })

  if (!backendRes.ok) {
    return new Response(
      JSON.stringify({ error: `Backend error: ${backendRes.status}` }),
      { status: backendRes.status, headers: { 'Content-Type': 'application/json' } }
    )
  }

  return new Response(backendRes.body, {
    headers: {
      'Content-Type': backendRes.headers.get('Content-Type') || 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  })
}