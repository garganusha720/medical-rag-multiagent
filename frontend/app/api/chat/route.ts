import { NextRequest } from 'next/server'

// API_URL is for server-side (Docker internal: http://backend:8000)
// NEXT_PUBLIC_API_URL is for client-side fallback (http://localhost:8000)
const BACKEND_URL = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function POST(req: NextRequest) {
  const body = await req.json()

  const token = req.headers.get('authorization')

  // First request can take 60-90s (BM25 index build + LLM calls)
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 300000) // 5 minute timeout

  try {
    const backendRes = await fetch(`${BACKEND_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: token } : {}),
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    clearTimeout(timeout)

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
  } catch (err) {
    clearTimeout(timeout)
    return new Response(
      JSON.stringify({ error: 'Backend request timed out or failed. Please try again.' }),
      { status: 504, headers: { 'Content-Type': 'application/json' } }
    )
  }
}