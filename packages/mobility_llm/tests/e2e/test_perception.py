"""
tests/e2e/test_perception.py — the perception_filter category, end to end, on a real gateway.

Requires a running gateway and provider keys: ``LLM_GATEWAY_E2E_URL=http://localhost:8000``.
Without the variable, the module is skipped (e2e marker). Still runnable as a script:

  python tests/e2e/test_perception.py --base-url http://staging:8000
"""
import argparse
import os
import sys
import time

import httpx
import pytest

E2E_URL = os.getenv("LLM_GATEWAY_E2E_URL")
if not E2E_URL and "pytest" in sys.modules and __name__ != "__main__":
    pytest.skip("LLM_GATEWAY_E2E_URL not set: no real gateway", allow_module_level=True)

DEFAULT_BASE_URL = "http://localhost:8000"
MIN_SUMMARY_LENGTH = 20

PAYLOAD = {
    "category": "perception_filter",
    "parameters": {},
    "agents": [
        {
            "agent_id": "ag_marc",
            "perception": "38y, Aeronautical Engineer, Pibrac.",
            "goal": "Travel time reliability to avoid unpredictable traffic congestion.",
            "context": "Daily home-work commute to city center; car used only for weekend leisure.",
            "constraints": "Strict dependence on SNCF schedules and connections at Matabiau station.",
            "history": [
                "Accident on the line: 1h delay, impossible to warn in time, day ruined."
            ],
            "feeling": "Mode Private Car: Cost: 0.6, Time: 0.3, Ease: 0.5, Safety: 0.9, Comfort: 0.9, Ecology: 0.4\nMode Train: Cost: 0.8, Time: 0.9, Ease: 0.8, Safety: 0.9, Comfort: 0.8, Ecology: 0.9"
        },
        {
            "agent_id": "ag_sarah",
            "perception": "21y, Master's student, Borderouge.",
            "goal": "Radical minimization of travel costs.",
            "context": "Trips to Rangueil campus and Jean Jaurès; grocery shopping on foot.",
            "constraints": "No personal vehicle or license; budget limited to Tisséo social fares.",
            "history": [
                "Bus late, had to run 15 minutes in the rain so as not to miss the exam."
            ],
            "feeling": "Mode Walking: Cost: 1.0, Time: 0.6, Ease: 0.9, Safety: 0.7, Comfort: 0.6, Ecology: 1.0\nMode Bus / Metro: Cost: 1.0, Time: 0.9, Ease: 1.0, Safety: 0.8, Comfort: 0.7, Ecology: 0.9"
        },
        {
            "agent_id": "ag_thomas",
            "perception": "38y, Private Banker, lives in Vieille-Toulouse.",
            "goal": "Reflect professional success and enjoy a high-end sensory experience during commutes.",
            "context": "Primarily a direct commute between home and the city center or Labège.",
            "constraints": "None",
            "history": [
                "Massive traffic jam: 50 min trip, max stress, arrived late and annoyed.",
                "Smooth trip: 18 min, air conditioning on, arrived fresh and rested."
            ],
            "feeling": "Mode Walking: Cost: Perfect, Time: Very Good, Ease: Perfect, Safety: Very Good, Comfort: Good, Ecology: Perfect\nMode Private Car: Cost: Fair, Time: Unsatisfactory, Ease: Average, Safety: Excellent, Comfort: Excellent, Ecology: Subpar"
        }
    ]
}


def main(base_url: str) -> None:
    """Raises AssertionError on failure (pytest); the script turns it into an exit code."""
    print(f"\n{'═' * 60}")
    print("  TEST PERCEPTION FILTER")
    print(f"  {base_url}")
    print(f"{'═' * 60}")
    t0 = time.monotonic()
    max_latency_ms = 20_000

    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{base_url}/tasks", json=PAYLOAD)
        if resp.status_code != 202:
            print(f"  ✗ Submission error (HTTP {resp.status_code}): {resp.text}")
            raise AssertionError('perception_filter scenario failed')

        task_id = resp.json()["task_id"]
        print(f"  → Task {task_id} accepted. Waiting...")

        for attempt in range(30):
            time.sleep(1.5)
            res = client.get(f"{base_url}/tasks/{task_id}").json()
            status = res["status"]
            print(f"  [poll #{attempt + 1}] Status: {status}...")

            if status == "success":
                elapsed = time.monotonic() - t0
                print(f"\n  ✓ HTTP success! Finished in {elapsed:.2f}s")

                # 1. Performance check
                latency_ms = res.get("latency_ms", 0)
                if latency_ms > max_latency_ms:
                    print(f"  ⚠ High LLM latency ({latency_ms:.0f}ms > {max_latency_ms}ms)")
                else:
                    print(f"  ✓ Perfo OK ({latency_ms:.0f}ms)")

                # 2. Content validation
                results = res.get("result", [])
                if not results:
                    print("  ✗ Empty result!")
                    raise AssertionError('perception_filter scenario failed')

                all_ok = True
                for agent in results:
                    summary = agent.get("summary", "")
                    print(f"    - Agent ID : {agent.get('agent_id')}")
                    print(f"    - Summary  : {summary}")
                    if not summary or len(summary) < MIN_SUMMARY_LENGTH:
                        print(f"  ✗ 'summary' empty or too short (< {MIN_SUMMARY_LENGTH} chars)!")
                        all_ok = False

                if not all_ok:
                    raise AssertionError('perception_filter scenario failed')

                print("\n  ✓ All checks (performance & validation) passed.")
                return

            elif status == "failed":
                print(f"\n  ✗ Task failed: {res.get('error')}")
                raise AssertionError('perception_filter scenario failed')

        print("\n  ✗ Timeout expired before the end of the task.")
        raise AssertionError('perception_filter scenario failed')


def test_perception_filter_de_bout_en_bout():
    main(E2E_URL)


if __name__ == "__main__":  # pragma: no cover - usage script
    parser = argparse.ArgumentParser(description="Test perception filter end-to-end")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()
    main(args.base_url)
