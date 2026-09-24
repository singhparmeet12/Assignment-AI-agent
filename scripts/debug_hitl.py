"""
scripts/debug_hitl.py: Diagnostic test script for Human-in-the-Loop workflow.
Directly calls run_newsletter_agent(goal, mode="human_in_loop") and prints the returned status.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import run_newsletter_agent

def main():
    print("\n" + "=" * 60)
    print("Testing stream_newsletter_agent(mode='human_in_loop')")
    print("=" * 60)
    from agent.graph import stream_newsletter_agent
    goal = "Create a short test briefing on latest autonomous AI agent frameworks."
    final_resp = None
    for node_name, update, data in stream_newsletter_agent(goal=goal, mode="human_in_loop"):
        print(f"YIELDED node_name: {node_name}")
        if node_name in ["complete", "interrupt"]:
            final_resp = data
            print(f"Captured final_response with status: {final_resp.get('status')}")

    print("=" * 60)
    print(f"STREAM FINAL RESPONSE STATUS: {final_resp.get('status') if final_resp else 'None'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
