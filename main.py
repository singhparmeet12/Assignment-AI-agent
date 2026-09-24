"""
main.py: CLI Entry Point for the Newsletter Agent.

Provides a rich command-line interface to execute the LangGraph Newsletter Agent
without the web UI. Supports fully autonomous execution and interactive terminal
Human-in-the-Loop (HITL) review.
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from agent.graph import run_newsletter_agent, resume_newsletter_agent


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the AI Newsletter Agent from the command line."
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Create a weekly newsletter on latest AI agent news and send it to our subscribers.",
        help="Target goal, topic, or audience for the newsletter."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["autonomous", "hitl", "human_in_loop"],
        default="autonomous",
        help="Execution mode: 'autonomous' for hands-off execution, or 'hitl' for terminal review."
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Maximum self-critique loop revisions (default: 2)."
    )
    return parser.parse_args()


def print_stage_logs(stage_log: list[str]):
    print("\n📋 Execution Log:")
    print("-" * 60)
    for idx, entry in enumerate(stage_log, 1):
        if "warning" in entry.lower() or "fail" in entry.lower():
            print(f"  ⚠️  [{idx}] {entry}")
        elif "complete" in entry.lower() or "passed" in entry.lower() or "published" in entry.lower():
            print(f"  ✅ [{idx}] {entry}")
        else:
            print(f"  ℹ️  [{idx}] {entry}")
    print("-" * 60)


def main():
    args = parse_arguments()
    mode = "human_in_loop" if args.mode == "hitl" else args.mode

    print("=" * 72)
    print("                📰 AI NEWSLETTER AGENT — CLI RUNNER")
    print("=" * 72)
    print(f"Goal:           {args.goal}")
    print(f"Mode:           {mode}")
    print(f"Max Revisions:  {args.max_revisions}")
    print("=" * 72)
    print("\n🚀 Starting agent execution...")

    try:
        response = run_newsletter_agent(
            goal=args.goal,
            mode=mode,
        )
    except Exception as exc:
        print(f"\n❌ Execution aborted due to error:\n{exc}")
        sys.exit(1)

    status = response.get("status")
    state = response.get("state", {})
    thread_id = response.get("thread_id")

    # Handle Human-in-the-Loop Interrupt
    while status == "interrupted":
        print_stage_logs(state.get("stage_log", []))
        print("\n" + "=" * 72)
        print("          ⏸️  HUMAN-IN-THE-LOOP EDITORIAL CHECKPOINT")
        print("=" * 72)
        print(f"\nSubject Line:  {state.get('subject_line')}")
        critique = state.get("critique", {})
        print(f"Critic Score:  {critique.get('score', 'N/A')}/10 (Passed: {critique.get('passed', False)})")
        print("\nCurrent Draft Markdown Preview:")
        print("-" * 60)
        draft = state.get("draft_markdown", "")
        preview_lines = draft.splitlines()[:25]
        print("\n".join(preview_lines))
        if len(draft.splitlines()) > 25:
            print(f"\n... [{len(draft.splitlines()) - 25} more lines] ...")
        print("-" * 60)

        choice = input("\nDo you approve this draft for publication? (y/n): ").strip().lower()
        if choice in ["y", "yes"]:
            print("\n✅ Approved! Resuming graph to publish...")
            response = resume_newsletter_agent(
                thread_id=thread_id,
                decision="approve"
            )
        else:
            feedback = input("Enter your editorial feedback / requested changes: ").strip()
            print(f"\n📝 Requesting revision with feedback: '{feedback}'")
            response = resume_newsletter_agent(
                thread_id=thread_id,
                decision="request_changes",
                feedback=feedback
            )

        status = response.get("status")
        state = response.get("state", {})

    # Final Outcome
    print_stage_logs(state.get("stage_log", []))

    if status == "completed":
        print("\n🎉 Newsletter generation completed successfully!")
        print(f"Subject: {state.get('subject_line')}")
        print("\nSaved Deliverables:")
        print("  - outputs/newsletter.md")
        print("  - outputs/newsletter.html\n")
    else:
        print(f"\n⚠️  Finished with status: {status}")


if __name__ == "__main__":
    main()
