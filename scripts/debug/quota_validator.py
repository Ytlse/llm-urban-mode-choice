#!/usr/bin/env python3
"""
Validates RPM/TPM limits on each separate minute (60s sliding windows).
For each provider, checks that quotas are never exceeded,
otherwise diagnoses the violated thresholds and problematic timestamps.
Also counts cooldowns (5xx/429 errors).
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

# Providers configuration
PROVIDERS_CONFIG = {
    "google_gemini31_key1": {"rpm_limit": 15, "tpm_limit": 250000},
    "google_gemini35_key1": {"rpm_limit": 15, "tpm_limit": 250000},
    "google_gemma42_key1": {"rpm_limit": 15, "tpm_limit": float('inf')},
    "google_gemma43_key1": {"rpm_limit": 15, "tpm_limit": float('inf')},
    "groq_llama3": {"rpm_limit": 30, "tpm_limit": 12000},
    "groq_llama4": {"rpm_limit": 30, "tpm_limit": 30000},
    "groq_qwen": {"rpm_limit": 2, "tpm_limit": 6000},
    "groq_llama31": {"rpm_limit": 2, "tpm_limit": 6000},
    "groq_openai_120_key1": {"rpm_limit": 30, "tpm_limit": 8000},
    "groq_openai_20": {"rpm_limit": 30, "tpm_limit": 8000},
    "cerebras_gptoss120b_key1": {"rpm_limit": 5, "tpm_limit": 30000},
    "cerebras_zai-glm-4.7": {"rpm_limit": 5, "tpm_limit": 30000},
    "openai": {"rpm_limit": 15, "tpm_limit": 200000},
    "mistral": {"rpm_limit": 90, "tpm_limit": 500000},
}


def parse_timestamp(ts_str):
    """Parse ISO format timestamp."""
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    except:
        return None


def load_llm_exchanges(run_dir):
    """Load llm_exchanges.jsonl from run directory (supports pretty-printed JSON)."""
    log_file = Path(run_dir) / "llm_exchanges.jsonl"
    if not log_file.exists():
        print(f"❌ File not found: {log_file}")
        return []

    exchanges = []
    with open(log_file, "r") as f:
        content = f.read()

    # Parse multi-line JSON objects
    current_obj = ""
    depth = 0

    for char in content:
        current_obj += char
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(current_obj.strip())
                    exchanges.append(obj)
                except json.JSONDecodeError:
                    pass
                current_obj = ""

    return exchanges


def count_cooldowns(exchanges):
    """Counts cooldowns per provider (5xx and 429 errors that trigger disable_timeout)."""
    cooldowns = defaultdict(int)

    for exchange in exchanges:
        provider = exchange.get("provider")
        if not provider:
            continue

        # A cooldown is triggered by an HTTP error
        error = exchange.get("error") or exchange.get("error_message")
        status = exchange.get("http_status")

        # Cooldown on 5xx (server errors) and 429 (rate limit)
        if error and any(code in str(error) for code in ["500", "503", "429"]):
            cooldowns[provider] += 1
        elif status and str(status) in ["429", "500", "503"]:
            cooldowns[provider] += 1

    return dict(cooldowns)


def calculate_batch_metrics(exchanges, provider):
    """Computes batching statistics for a provider."""
    provider_exchanges = [
        e for e in exchanges
        if e.get("provider") == provider and e.get("tokens_in") is not None
    ]

    if not provider_exchanges:
        return None

    # Group by task_id (batch)
    batches = {}
    for ex in provider_exchanges:
        task_id = ex.get("task_id", "unknown")
        if task_id not in batches:
            batches[task_id] = []
        batches[task_id].append(ex)

    batch_sizes = [len(batch) for batch in batches.values()]

    if not batch_sizes:
        return None

    return {
        "batch_count": len(batches),
        "total_exchanges": len(provider_exchanges),
        "batch_min": min(batch_sizes),
        "batch_max": max(batch_sizes),
        "batch_mean": statistics.mean(batch_sizes),
        "batch_median": statistics.median(batch_sizes),
        "utilization": (len(provider_exchanges) / len(batches)) if len(batches) > 0 else 0,
    }


def calculate_sliding_window_metrics(exchanges, provider, window_seconds=60):
    """
    Computes RPM/TPM metrics over 60s sliding windows.
    Returns: (violations, metrics_by_minute)
    """
    config = PROVIDERS_CONFIG.get(provider, {"rpm_limit": float('inf'), "tpm_limit": float('inf')})
    rpm_limit = config["rpm_limit"]
    tpm_limit = config["tpm_limit"]

    # Filter the exchanges for this provider
    provider_exchanges = [
        e for e in exchanges
        if e.get("provider") == provider and e.get("tokens_in") is not None
    ]

    if not provider_exchanges:
        return [], {}

    # Sort by timestamp
    provider_exchanges.sort(key=lambda x: x.get("time", ""))

    violations = []
    metrics_by_minute = {}

    # Build a complete timeline
    timestamps = [parse_timestamp(e.get("time", "")) for e in provider_exchanges]
    timestamps = [t for t in timestamps if t]

    if not timestamps:
        return [], {}

    min_time = timestamps[0]
    max_time = timestamps[-1]

    # 1-minute windows (whole minute)
    current_time = min_time.replace(second=0, microsecond=0)

    while current_time <= max_time:
        window_start = current_time
        window_end = current_time + timedelta(seconds=59, microseconds=999999)

        # Fetch all exchanges in this window
        window_exchanges = [
            e for e, ts in zip(provider_exchanges, timestamps)
            if window_start <= ts <= window_end
        ]

        if window_exchanges:
            rpm_count = len(window_exchanges)
            tpm_count = sum(e.get("tokens_in", 0) + e.get("tokens_out", 0) for e in window_exchanges)

            minute_key = window_start.strftime("%Y-%m-%d %H:%M")
            metrics_by_minute[minute_key] = {
                "rpm": rpm_count,
                "tpm": tpm_count,
                "rpm_limit": rpm_limit,
                "tpm_limit": tpm_limit,
                "exchanges": len(window_exchanges),
                "timestamp": window_start.isoformat(),
            }

            # Violation detection
            if rpm_count > rpm_limit:
                violations.append({
                    "type": "RPM",
                    "minute": minute_key,
                    "value": rpm_count,
                    "limit": rpm_limit,
                    "excess": rpm_count - rpm_limit,
                })

            if tpm_limit != float('inf') and tpm_count > tpm_limit:
                violations.append({
                    "type": "TPM",
                    "minute": minute_key,
                    "value": tpm_count,
                    "limit": tpm_limit,
                    "excess": tpm_count - tpm_limit,
                })

        current_time += timedelta(minutes=1)

    return violations, metrics_by_minute


def generate_report(run_dir):
    """Generates the quota validation report."""
    print("=" * 90)
    print("📊 RPM/TPM QUOTA VALIDATION (60s sliding windows)")
    print("=" * 90)

    exchanges = load_llm_exchanges(run_dir)

    if not exchanges:
        print("❌ No LLM exchange found")
        return

    print(f"✅ {len(exchanges)} LLM exchanges loaded\n")

    # Group by provider
    providers_in_log = set(e.get("provider") for e in exchanges if e.get("provider"))
    print(f"Providers in the logs: {sorted(providers_in_log)}\n")

    # Count the cooldowns
    cooldowns = count_cooldowns(exchanges)
    total_cooldowns = sum(cooldowns.values())
    if total_cooldowns > 0:
        print(f"⚠️  Cooldowns detected: {total_cooldowns} total")
        for provider, count in sorted(cooldowns.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   {provider}: {count}")
        print()

    all_violations = {}
    all_metrics = {}

    for provider in sorted(providers_in_log):
        if provider not in PROVIDERS_CONFIG:
            continue

        violations, metrics = calculate_sliding_window_metrics(exchanges, provider)
        all_violations[provider] = violations
        all_metrics[provider] = metrics

        config = PROVIDERS_CONFIG[provider]
        rpm_limit = config["rpm_limit"]
        tpm_limit = config["tpm_limit"]

        print(f"\n{'='*90}")
        print(f"🔹 Provider: {provider}")
        print(f"   Limits: RPM={rpm_limit}, TPM={'∞' if tpm_limit == float('inf') else tpm_limit}")

        if not metrics:
            print(f"   ⚠️  No exchange")
            continue

        # Stats
        rpm_values = [m["rpm"] for m in metrics.values()]
        tpm_values = [m["tpm"] for m in metrics.values()]

        print(f"\n   RPM statistics (requests/minute):")
        print(f"     • Min: {min(rpm_values)}, Max: {max(rpm_values)}, Avg: {statistics.mean(rpm_values):.1f}")

        print(f"\n   TPM statistics (tokens/minute):")
        if tpm_limit == float('inf'):
            print(f"     • Min: {min(tpm_values)}, Max: {max(tpm_values)}, Avg: {statistics.mean(tpm_values):.1f}")
            print(f"     • Unlimited quota (no TPM cap)")
        else:
            print(f"     • Min: {min(tpm_values)}, Max: {max(tpm_values)}, Avg: {statistics.mean(tpm_values):.1f}")

        if cooldowns.get(provider, 0) > 0:
            print(f"\n   ⚠️  Cooldowns: {cooldowns[provider]} (provider benched {cooldowns[provider]}x)")

        if violations:
            print(f"\n   🚨 VIOLATIONS DETECTED: {len(violations)}")

            for v in violations[:5]:  # Top 5
                print(f"     • {v['type']:3s} @ {v['minute']}: {v['value']:5d} / {v['limit']:5d} (excess: +{v['excess']})")

            if len(violations) > 5:
                print(f"     ... and {len(violations) - 5} more violations")
        else:
            print(f"   ✅ NO VIOLATION (quotas respected)")

    # Full summary table
    print(f"\n{'='*90}")
    print("📊 TABLEAU RÉCAPITULATIF - TOUS LES PROVIDERS")
    print(f"{'='*90}\n")

    # Prepare the data
    table_data = []
    for provider in sorted(providers_in_log):
        if provider not in PROVIDERS_CONFIG:
            continue

        config = PROVIDERS_CONFIG[provider]
        rpm_limit = config["rpm_limit"]
        tpm_limit = config["tpm_limit"]
        metrics = all_metrics.get(provider, {})
        violations = all_violations.get(provider, [])
        batch_metrics = calculate_batch_metrics(exchanges, provider)

        if not metrics:
            table_data.append({
                "provider": provider,
                "rpm_max": 0,
                "tpm_max": 0,
                "rpm_mean": 0,
                "tpm_mean": 0,
                "batch_count": 0,
                "batch_mean": 0,
                "cooldown_count": 0,
                "violations": "—",
            })
            continue

        rpm_values = [m["rpm"] for m in metrics.values()]
        tpm_values = [m["tpm"] for m in metrics.values()]

        rpm_max = max(rpm_values)
        tpm_max = max(tpm_values)
        rpm_mean = statistics.mean(rpm_values)
        tpm_mean = statistics.mean(tpm_values)

        batch_count = batch_metrics.get("batch_count", 0) if batch_metrics else 0
        batch_mean = batch_metrics.get("batch_mean", 0) if batch_metrics else 0

        violation_str = "✅" if not violations else f"🚨 {len(violations)}"
        cooldown_count = cooldowns.get(provider, 0)

        table_data.append({
            "provider": provider,
            "rpm_max": rpm_max,
            "rpm_limit": rpm_limit,
            "tpm_max": tpm_max,
            "tpm_limit": tpm_limit,
            "rpm_mean": rpm_mean,
            "tpm_mean": tpm_mean,
            "batch_count": batch_count,
            "batch_mean": batch_mean,
            "cooldown_count": cooldown_count,
            "violations": violation_str,
        })

    # Print the table
    print(f"{'Provider':<25} {'RPM':<16} {'TPM':<20} {'Batch':<12} {'Cooldown':<12} {'Violations':<12}")
    print(f"{'':<25} {'Max/L/Moy':<16} {'Max/L/Moy':<20} {'Nbre/Moy':<12} {'':<12} {'':<12}")
    print("─" * 127)

    for row in table_data:
        provider = row["provider"][:24]

        rpm_info = f"{row['rpm_max']}/{row['rpm_limit']}/{row['rpm_mean']:.1f}"

        if row["tpm_limit"] == float('inf'):
            tpm_info = f"{row['tpm_max']}/∞/{row['tpm_mean']:.0f}"
        else:
            tpm_info = f"{row['tpm_max']}/{row['tpm_limit']}/{row['tpm_mean']:.0f}"

        batch_info = f"{row['batch_count']}/{row['batch_mean']:.1f}"
        cooldown_info = f"{row['cooldown_count']}" if row['cooldown_count'] > 0 else "—"
        violations_str = row["violations"]

        print(f"{provider:<25} {rpm_info:<16} {tpm_info:<20} {batch_info:<12} {cooldown_info:<12} {violations_str:<12}")

    print()

    # Global summary
    print(f"\n{'='*90}")
    print("📋 GLOBAL SUMMARY")
    print(f"{'='*90}")

    total_violations = sum(len(v) for v in all_violations.values())
    providers_violated = [p for p, v in all_violations.items() if v]

    if total_violations == 0:
        print("✅ SUCCÈS: Aucune violation de quota détectée")
    else:
        print(f"🚨 ALERTE: {total_violations} violations détectées")

    if total_cooldowns > 0:
        print(f"⚠️  {total_cooldowns} cooldowns detected (errors 5xx/429)")

    print()


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/current"
    generate_report(run_dir)
