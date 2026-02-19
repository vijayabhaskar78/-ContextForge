"""
Performance metrics collector for the ContextForge API service.

Tracks: latency, throughput, memory usage, cost per query, error count.
Logs structured metrics to logs/metrics.jsonl.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import psutil


# ---------------------------------------------------------------------------
# Groq pricing per million tokens (USD).  Update as needed.
# ---------------------------------------------------------------------------
_GROQ_PRICING: dict[str, dict[str, float]] = {
    "qwen/qwen3-32b":   {"input": 0.29,  "output": 0.39},
    "gemma2-9b-it":      {"input": 0.20,  "output": 0.20},
    "llama-3.3-70b":     {"input": 0.59,  "output": 0.79},
    "llama-3.1-8b":      {"input": 0.05,  "output": 0.08},
    "_default":          {"input": 0.30,  "output": 0.40},
}


def _get_pricing(model: str) -> dict[str, float]:
    return _GROQ_PRICING.get(model, _GROQ_PRICING["_default"])


class MetricsCollector:
    """Thread-safe request metrics tracker with JSONL file logging."""

    def __init__(self, log_dir: str | Path = "logs"):
        self._lock = threading.Lock()
        self._start_time: float = time.time()

        # Counters.
        self._total_requests: int = 0
        self._total_latency_ms: float = 0.0
        self._error_count: int = 0
        self._min_latency_ms: float = float("inf")
        self._max_latency_ms: float = 0.0

        # Token / cost tracking.
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0

        # Logging.
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "metrics.jsonl"

        # Process handle for memory tracking.
        self._process = psutil.Process(os.getpid())

    def record_request(
        self,
        latency_ms: float,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
    ) -> None:
        """Records a single request's outcome and appends to JSONL log."""
        pricing = _get_pricing(model)
        cost_usd = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 8),
        }

        with self._lock:
            self._total_requests += 1
            self._total_latency_ms += latency_ms
            if latency_ms < self._min_latency_ms:
                self._min_latency_ms = latency_ms
            if latency_ms > self._max_latency_ms:
                self._max_latency_ms = latency_ms
            if not success:
                self._error_count += 1
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cost_usd += cost_usd

        # Append to JSONL file outside the lock.
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except OSError:
            pass

    def get_summary(self) -> dict:
        """Returns comprehensive metrics snapshot."""
        with self._lock:
            total = self._total_requests
            avg_lat = (self._total_latency_ms / total) if total > 0 else 0.0
            min_lat = self._min_latency_ms if total > 0 else 0.0
            max_lat = self._max_latency_ms if total > 0 else 0.0
            errors = self._error_count
            in_tok = self._total_input_tokens
            out_tok = self._total_output_tokens
            cost = self._total_cost_usd

        # Throughput.
        uptime_s = time.time() - self._start_time
        throughput_rps = (total / uptime_s) if uptime_s > 0 else 0.0

        # Memory usage.
        mem_info = self._process.memory_info()
        mem_rss_mb = mem_info.rss / (1024 * 1024)
        mem_vms_mb = mem_info.vms / (1024 * 1024)

        # Cost.
        avg_cost = (cost / total) if total > 0 else 0.0

        return {
            "latency": {
                "avg_ms": round(avg_lat, 2),
                "min_ms": round(min_lat, 2),
                "max_ms": round(max_lat, 2),
            },
            "throughput": {
                "total_requests": total,
                "requests_per_second": round(throughput_rps, 4),
                "uptime_seconds": round(uptime_s, 1),
            },
            "memory": {
                "rss_mb": round(mem_rss_mb, 1),
                "vms_mb": round(mem_vms_mb, 1),
            },
            "cost": {
                "total_usd": round(cost, 6),
                "avg_per_query_usd": round(avg_cost, 6),
                "total_input_tokens": in_tok,
                "total_output_tokens": out_tok,
            },
            "errors": {
                "count": errors,
                "rate_percent": round((errors / total * 100) if total > 0 else 0.0, 2),
            },
        }


# Module-level singleton used by the API server.
metrics_collector = MetricsCollector()
