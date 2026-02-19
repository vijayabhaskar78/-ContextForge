"""
Load test for ContextForge API.
Sends concurrent requests to measure throughput and latency under load.

Usage:  python load_test.py [num_requests] [concurrency]
"""
import sys
import time
import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://127.0.0.1:8000/query"

QUERIES = [
    "What is RAG?",
    "What are the key principles of building AI agents?",
    "Explain the role of memory in AI agents.",
    "What tools do AI agents use?",
    "How do AI agents handle prompts?",
    "What is retrieval-augmented generation?",
    "Explain agentic workflows.",
    "What are providers in AI agent architecture?",
    "How does context affect AI agent performance?",
    "What is the role of models in AI agents?",
]


def send_query(query: str, session_id: str) -> dict:
    payload = json.dumps({"query": query, "session_id": session_id}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            latency = (time.perf_counter() - start) * 1000
            return {"success": True, "latency_ms": latency, "answer_len": len(data.get("answer", ""))}
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return {"success": False, "latency_ms": latency, "error": str(e)}


def main():
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print(f"\n{'='*60}")
    print(f"  ContextForge Load Test")
    print(f"  Requests: {num_requests}  |  Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    results = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(num_requests):
            q = QUERIES[i % len(QUERIES)]
            sid = f"load-test-{i}"
            futures.append(pool.submit(send_query, q, sid))

        for f in as_completed(futures):
            r = f.result()
            status = "OK" if r["success"] else "FAIL"
            print(f"  [{status}] {r['latency_ms']:>8.1f} ms")
            results.append(r)

    wall_elapsed = time.perf_counter() - wall_start

    # Summary.
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency_ms"] for r in successes]

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total requests:    {num_requests}")
    print(f"  Successful:        {len(successes)}")
    print(f"  Failed:            {len(failures)}")
    print(f"  Wall time:         {wall_elapsed:.2f} s")
    print(f"  Throughput:        {num_requests / wall_elapsed:.2f} req/s")
    if latencies:
        print(f"  Avg latency:       {sum(latencies)/len(latencies):.0f} ms")
        print(f"  Min latency:       {min(latencies):.0f} ms")
        print(f"  Max latency:       {max(latencies):.0f} ms")
        p50 = sorted(latencies)[len(latencies)//2]
        p95 = sorted(latencies)[int(len(latencies)*0.95)]
        print(f"  P50 latency:       {p50:.0f} ms")
        print(f"  P95 latency:       {p95:.0f} ms")
    print(f"  Error rate:        {len(failures)/num_requests*100:.1f}%")
    print(f"{'='*60}\n")

    # Fetch server metrics.
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/metrics")
        with urllib.request.urlopen(req) as resp:
            metrics = json.loads(resp.read())
            print("  Server /metrics snapshot:")
            print(json.dumps(metrics, indent=4))
    except Exception:
        pass


if __name__ == "__main__":
    main()
