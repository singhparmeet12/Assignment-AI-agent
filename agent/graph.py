"""
agent/graph.py: LangGraph Orchestration & Single Entry-Point.

================================================================================
                        NEWSLETTER AGENT GRAPH TOPOLOGY
================================================================================

                                    [START]
                                       │
                                       ▼
                                 [planner_node]
                                       │
                                       ▼
                       ┌──────► [researcher_node] ◄─────┐
                       │               │                │
                       │               ├────────────────┘ [retry_research:
                       │               │                   0 summaries & attempts < max]
                       │               ▼
    ┌──────────────────┴───────► [writer_node]
    │                                  │
    │                                  ▼
    │                            [critic_node]
    │                                  │
    │   [revise: not passed            ├─────── [publish: mode="autonomous"
    │    & rev < max_rev]              │         or max revisions reached]
    │                                  │                 │
    └─── [reviser_node] ◄──────────────┤                 │
             ▲                         │                 │
             │                         ▼                 │
             │           [human_review:                  │
             │            mode="human_in_loop"]          │
             │                         │                 │
             │                         ▼                 │
             │               [human_review_node]         │
             │               (interrupt / resume)        │
             │                         │                 │
             │   [revise: not approved │                 │
             └─── & rev < max_rev] ────┤                 │
                                       │                 │
                                       ▼                 │
                          [publish: approved             │
                           or rev >= max_rev]            │
                                       │                 │
                                       ├─────────────────┘
                                       ▼
                                [publisher_node]
                                       │
                                       ▼
                                     [END]

================================================================================
"""

import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Optional, Literal, Any
from dotenv import load_dotenv

# Ensure workspace root is in sys.path when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.state import NewsletterState, initial_state
from agent.nodes import (
    planner_node,
    researcher_node,
    writer_node,
    critic_node,
    reviser_node,
    human_review_node,
    publisher_node,
)

# Load environment variables
load_dotenv()

# Configure module logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Conditional Edge Routing Functions
# ==============================================================================

def should_continue_research(
    state: NewsletterState
) -> Literal["retry_research", "writer"]:
    """
    Autonomous Self-Correction Gate for Research:
    If research_summary is empty and research_attempts < max_research_attempts,
    routes back to researcher_node for automated retry with broader parameters.
    Otherwise, proceeds forward to writer_node.

    Args:
        state: Current NewsletterState.

    Returns:
        Literal["retry_research", "writer"]
    """
    research_summary = state.get("research_summary", [])
    research_attempts = state.get("research_attempts", 0)
    max_attempts = state.get("max_research_attempts", 3)
    research_failed = state.get("research_failed", False)

    if not research_summary and not research_failed and research_attempts < max_attempts:
        logger.info(
            f"Research returned 0 summaries. Triggering self-correction retry "
            f"(attempt {research_attempts + 1}/{max_attempts})..."
        )
        return "retry_research"

    return "writer"


def should_continue_revision(
    state: NewsletterState
) -> Literal["revise", "human_review", "publish"]:
    """
    Evaluates whether the draft requires another revision cycle or proceeds
    forward to human review (HITL mode) or direct publication (autonomous mode).

    Routing rules:
    - "revise": Critique failed AND revision_count < max_revisions -> routes to reviser -> writer.
    - "human_review": Critique passed (or max_revisions hit) AND mode == "human_in_loop".
    - "publish": Critique passed (or max_revisions hit) AND mode == "autonomous".

    Args:
        state: Current NewsletterState.

    Returns:
        Literal["revise", "human_review", "publish"]
    """
    critique = state.get("critique", {})
    passed = bool(critique.get("passed", False) or critique.get("pass", False))
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)
    mode = state.get("mode", "autonomous")

    logger.info(
        f"Evaluating should_continue_revision: passed={passed}, "
        f"revision_count={revision_count}/{max_revisions}, mode='{mode}'"
    )

    if not passed and revision_count < max_revisions:
        logger.info("Routing decision: 'revise' -> reviser_node")
        return "revise"

    if mode == "human_in_loop":
        logger.info("Routing decision: 'human_review' -> human_review_node")
        return "human_review"

    logger.info("Routing decision: 'publish' -> publisher_node")
    return "publish"


def should_continue_after_human(
    state: NewsletterState
) -> Literal["revise", "publish"]:
    """
    Evaluates next steps after human review checkpoint.

    Routing rules:
    - "revise": Human requested changes (approved is False) AND revision_count < max_revisions.
    - "publish": Human approved draft (approved is True) OR revision_count >= max_revisions.

    Args:
        state: Current NewsletterState.

    Returns:
        Literal["revise", "publish"]
    """
    approved = state.get("approved", False)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)

    logger.info(
        f"Evaluating should_continue_after_human: approved={approved}, "
        f"revision_count={revision_count}/{max_revisions}"
    )

    if not approved and revision_count < max_revisions:
        logger.info("Routing decision: 'revise' -> reviser_node")
        return "revise"

    logger.info("Routing decision: 'publish' -> publisher_node")
    return "publish"


# ==============================================================================
# Graph Assembly & Compilation
# ==============================================================================

def build_newsletter_graph(checkpointer: Optional[Any] = None):
    """
    Assembles the 7-node LangGraph StateGraph with self-critique cycles,
    human-in-the-loop interruption boundaries, and memory checkpointers.

    Args:
        checkpointer: Optional persistence checkpointer (default: MemorySaver).

    Returns:
        CompiledStateGraph: The compiled, runnable LangGraph agent.
    """
    builder = StateGraph(NewsletterState)

    # 1. Register all 7 nodes
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("reviser", reviser_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("publisher", publisher_node)

    # 2. Sequential start into planner and researcher
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")

    # 3. Autonomous self-correcting research retry loop
    builder.add_conditional_edges(
        "researcher",
        should_continue_research,
        {
            "retry_research": "researcher",
            "writer": "writer",
        }
    )

    builder.add_edge("writer", "critic")

    # 3. Conditional routing from critic_node
    builder.add_conditional_edges(
        "critic",
        should_continue_revision,
        {
            "revise": "reviser",
            "human_review": "human_review",
            "publish": "publisher",
        }
    )

    # 4. Revision loop back to writer_node
    builder.add_edge("reviser", "writer")

    # 5. Conditional routing from human_review_node
    builder.add_conditional_edges(
        "human_review",
        should_continue_after_human,
        {
            "revise": "reviser",
            "publish": "publisher",
        }
    )

    # 6. Terminal edge
    builder.add_edge("publisher", END)

    # 7. Compile with checkpointer for state persistence and interrupts
    active_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=active_checkpointer)


# Module-level default checkpointer and compiled graph instance
default_checkpointer = MemorySaver()
compiled_newsletter_graph = build_newsletter_graph(checkpointer=default_checkpointer)


# ==============================================================================
# Public API Entry Points
# ==============================================================================

def run_newsletter_agent(
    goal: str,
    mode: str = "autonomous",
    thread_id: Optional[str] = None
) -> dict:
    """
    Primary single entry point to execute the Newsletter Agent workflow.

    Args:
        goal: Target topic, subject, or editorial objective for the newsletter.
        mode: Execution mode, either 'autonomous' or 'human_in_loop'.
        thread_id: Unique thread identifier for session tracking and persistence.
                   A new UUID is generated if not provided.

    Returns:
        dict: Standardized outcome dictionary:
            - If paused for human review:
                {"status": "paused_for_review", "thread_id": str, "payload": dict, "state": dict}
            - If completed:
                {"status": "complete", "thread_id": str, "result": dict}
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    starting_state = initial_state(goal=goal, mode=mode)
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Starting run_newsletter_agent [thread_id={thread_id}, mode={mode}]...")
    result = compiled_newsletter_graph.invoke(starting_state, config=config)

    # Check for direct LangGraph interruption in returned dict
    if isinstance(result, dict) and "__interrupt__" in result and result["__interrupt__"]:
        interrupt_entry = result["__interrupt__"][0]
        payload = getattr(interrupt_entry, "value", interrupt_entry)
        clean_state = {k: v for k, v in result.items() if not k.startswith("__")}
        return {
            "status": "paused_for_review",
            "thread_id": thread_id,
            "payload": payload,
            "state": clean_state,
        }

    # Inspect checkpointer task queue for interrupts
    graph_state = compiled_newsletter_graph.get_state(config)
    if graph_state.tasks and any(t.interrupts for t in graph_state.tasks):
        interrupt_items = [i for t in graph_state.tasks for i in t.interrupts]
        if interrupt_items:
            payload = interrupt_items[0].value
            return {
                "status": "paused_for_review",
                "thread_id": thread_id,
                "payload": payload,
                "state": graph_state.values,
            }

    # Graph completed execution
    clean_result = {k: v for k, v in result.items() if not k.startswith("__")}
    return {
        "status": "complete",
        "thread_id": thread_id,
        "result": clean_result,
    }


def stream_newsletter_agent(
    goal: str,
    mode: str = "autonomous",
    thread_id: Optional[str] = None
):
    """
    Executes the Newsletter Agent and yields (node_name, state_update, full_response)
    at every stage so frontends (Streamlit) can display live progress in real-time.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())

    starting_state = initial_state(goal=goal, mode=mode)
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"Starting stream_newsletter_agent [thread_id={thread_id}, mode={mode}]...")
    current_state = dict(starting_state)

    for chunk in compiled_newsletter_graph.stream(starting_state, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if isinstance(update, dict):
                current_state.update(update)
            yield (node_name, update, current_state)

    # Check for interrupts in tasks or state
    graph_state = compiled_newsletter_graph.get_state(config)
    if graph_state.tasks and any(t.interrupts for t in graph_state.tasks):
        interrupt_items = [i for t in graph_state.tasks for i in t.interrupts]
        if interrupt_items:
            payload = interrupt_items[0].value
            yield (
                "interrupt",
                payload,
                {
                    "status": "paused_for_review",
                    "thread_id": thread_id,
                    "payload": payload,
                    "state": graph_state.values,
                }
            )
            return

    clean_result = {k: v for k, v in graph_state.values.items() if not k.startswith("__")}
    yield (
        "complete",
        clean_result,
        {
            "status": "complete",
            "thread_id": thread_id,
            "result": clean_result,
        }
    )


def resume_newsletter_agent(
    thread_id: str,
    decision: dict
) -> dict:
    """
    Resumes a paused Human-in-the-Loop agent execution by providing editorial feedback.

    Args:
        thread_id: The thread identifier associated with the paused workflow session.
        decision: Editorial decision payload, e.g.:
                  {"decision": "approve"}
                  or
                  {"decision": "request_changes", "feedback": "Expand on enterprise adoption"}

    Returns:
        dict: Standardized outcome dictionary:
            - If paused again: {"status": "paused_for_review", ...}
            - If completed:    {"status": "complete", "thread_id": str, "result": dict}
    """
    if not thread_id:
        raise ValueError("A valid thread_id is required to resume agent execution.")

    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Resuming agent for thread_id={thread_id} with decision: {decision}...")

    # Resume graph execution by supplying the Command object with the human decision
    result = compiled_newsletter_graph.invoke(Command(resume=decision), config=config)

    # Check for another interrupt
    if isinstance(result, dict) and "__interrupt__" in result and result["__interrupt__"]:
        interrupt_entry = result["__interrupt__"][0]
        payload = getattr(interrupt_entry, "value", interrupt_entry)
        clean_state = {k: v for k, v in result.items() if not k.startswith("__")}
        return {
            "status": "paused_for_review",
            "thread_id": thread_id,
            "payload": payload,
            "state": clean_state,
        }

    graph_state = compiled_newsletter_graph.get_state(config)
    if graph_state.tasks and any(t.interrupts for t in graph_state.tasks):
        interrupt_items = [i for t in graph_state.tasks for i in t.interrupts]
        if interrupt_items:
            payload = interrupt_items[0].value
            return {
                "status": "paused_for_review",
                "thread_id": thread_id,
                "payload": payload,
                "state": graph_state.values,
            }

    clean_result = {k: v for k, v in result.items() if not k.startswith("__")}
    return {
        "status": "complete",
        "thread_id": thread_id,
        "result": clean_result,
    }


# ==============================================================================
# Manual Verification Block
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Newsletter Agent Workflow in Autonomous Mode")
    print("=" * 70)

    test_goal = "Weekly briefing on autonomous AI agents and open-source multi-agent frameworks"
    response = run_newsletter_agent(goal=test_goal, mode="autonomous")

    print(f"\nExecution Status: {response['status']}")
    print(f"Thread ID: {response['thread_id']}")

    if response["status"] == "complete":
        final_state = response["result"]
        print("\n--- Final Stage Log ---")
        for idx, entry in enumerate(final_state.get("stage_log", []), 1):
            print(f"[{idx}] {entry}")

        print("\n--- Deliverables ---")
        print(f"Subject Line:   {final_state.get('subject_line')}")
        print(f"Output File:    {final_state.get('output_path')}")
        print(f"Revision Count: {final_state.get('revision_count')}")
        print(f"Approved:       {final_state.get('approved')}")
    else:
        print("Agent execution paused or requires intervention:", response)
