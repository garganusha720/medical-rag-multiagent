"""
Interactive CLI Tester for Medical RAG API
Usage:
    python ask.py "What are the common symptoms of Long COVID?"
"""

import sys
import requests
import json

def ask(query: str):
    url = "http://localhost:8000/chat"
    headers = {"Content-Type": "application/json"}
    payload = {"query": query}

    print(f"\n[QUERY] {query}")
    print("=" * 60)
    print("Response:\n")

    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=120) as r:
            if r.status_code != 200:
                print(f"Error {r.status_code}: {r.text}")
                return

            final_metadata = None
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("0:"):
                        # Streamed token
                        token = json.loads(line[2:])
                        print(token, end="", flush=True)
                    elif line.startswith("2:"):
                        # Final metadata
                        meta_list = json.loads(line[2:])
                        if meta_list:
                            final_metadata = meta_list[0]

            print("\n\n" + "=" * 60)
            if final_metadata:
                citations = final_metadata.get("citations", [])
                coverage = final_metadata.get("citation_coverage", 0.0)
                session_id = final_metadata.get("session_id", "")
                print(f"Citation Coverage: {coverage:.0%}")
                print(f"Session ID: {session_id}")
                print(f"\nCitations ({len(citations)} verified sources):")
                for c in citations:
                    marker = c.get("marker", "")
                    src = c.get("source", "")
                    url_str = c.get("url", "")
                    txt = c.get("text", "")[:120] + "..."
                    print(f"  [{marker}] ({src}) {txt}")
                    if url_str:
                        print(f"       Link: {url_str}")
            print("=" * 60 + "\n")
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Could not connect to http://localhost:8000.")
        print("Make sure your FastAPI server is running in another terminal:")
        print("  cd D:\\MedicalLLM")
        print("  .venv\\Scripts\\activate")
        print("  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload\n")
    except requests.exceptions.Timeout:
        print("\n[ERROR] Request timed out after 120 seconds.")
        print("Check Terminal 1 for error messages.\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the common symptoms of Long COVID?"
    ask(query)
