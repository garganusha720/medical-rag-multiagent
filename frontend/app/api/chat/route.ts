import { NextRequest } from 'next/server'

// Mock streaming /chat endpoint — matches docs/api-contract.md
// Replace with real backend URL once Person A's FastAPI is ready

const MOCK_ANSWER =
  "Long COVID treatments focus on managing individual symptoms since there's no single cure yet [c1]. Common approaches include graded exercise therapy for fatigue, cognitive behavioral strategies for brain fog, and medications for specific symptoms like headaches or sleep issues [c2]. Some patients benefit from pulmonary rehabilitation if breathing problems persist [c3]. Always consult a healthcare provider for a personalized treatment plan."

const MOCK_CITATIONS = [
  {
    marker: 'c1',
    chunk_id: 'medlineplus_014_2',
    source: 'medlineplus',
    text: 'Long COVID refers to a wide range of symptoms that can last weeks or months after infection...',
    url: 'https://medlineplus.gov/longcovid.html',
    score: 5,
  },
  {
    marker: 'c2',
    chunk_id: 'pubmed_042_1',
    source: 'pubmed',
    text: 'Symptom-directed management remains the primary approach for Long COVID patients...',
    url: 'https://pubmed.ncbi.nlm.nih.gov/example',
    score: 4,
  },
  {
    marker: 'c3',
    chunk_id: 'medquad_008_1',
    source: 'medquad',
    text: 'Pulmonary rehabilitation programs can help patients with persistent respiratory symptoms...',
    url: 'https://medlineplus.gov/pulmonaryrehab.html',
    score: 4,
  },
]

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export async function POST(req: NextRequest) {
  const body = await req.json()
  const { session_id } = body

  const encoder = new TextEncoder()
  const words = MOCK_ANSWER.split(' ')

  const stream = new ReadableStream({
    async start(controller) {
      // Stream the answer word by word to simulate real token streaming
      for (const word of words) {
        controller.enqueue(encoder.encode(word + ' '))
        await sleep(40)
      }

      // Send the final JSON payload as a trailing chunk
      const finalPayload = {
        answer: MOCK_ANSWER,
        citations: MOCK_CITATIONS,
        citation_coverage: 0.85,
        session_id: session_id || crypto.randomUUID(),
      }
      controller.enqueue(
        encoder.encode('\n__FINAL__' + JSON.stringify(finalPayload))
      )
      controller.close()
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}