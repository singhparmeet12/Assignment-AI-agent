"""
agent/state.py: State Definition for the Newsletter Agent Graph.

This module defines the primary state schema (`NewsletterState`) for the LangGraph
newsletter workflow, tracking the progression from user goal to research, drafting,
self-critique/revision loops, human feedback, and final responsive publishing.
"""

from typing import TypedDict, Literal, Any


class ResearchArticle(TypedDict):
    """Structured representation of a synthesized research article/fact."""
    title: str
    summary: str
    source_url: str
    published_date: str


# Typed representation for critique result payload
CritiqueDict = TypedDict(
    "CritiqueDict",
    {
        "pass": bool,
        "issues": list[str],
        "suggestions": str,
    }
)


class NewsletterState(TypedDict):
    """
    Complete state dictionary passed across all nodes in the LangGraph newsletter workflow.
    """

    # goal:
    # - Writes: User / initial_state
    # - Reads: planner_node, writer_node, critic_node
    # - Why: Anchors the entire workflow to the user's original objective and editorial constraints.
    goal: str

    # mode:
    # - Writes: User / initial_state
    # - Reads: should_continue_revision (routing edge), human_review_node, critic_node
    # - Why: Selects workflow path: fully autonomous vs. human-in-the-loop interruption.
    mode: Literal["autonomous", "human_in_loop"]

    # plan:
    # - Writes: planner_node
    # - Reads: researcher_node, writer_node, critic_node
    # - Why: Stores the high-level research plan, narrative structure, and section outlines.
    plan: str

    # search_queries:
    # - Writes: planner_node
    # - Reads: researcher_node
    # - Why: Holds targeted search queries generated from the editorial plan for Tavily retrieval.
    search_queries: list[str]

    # raw_search_results:
    # - Writes: researcher_node
    # - Reads: researcher_node (internal/summarizer), debugging & audit logs
    # - Why: Retains raw payloads returned by external search tools for provenance and grounding.
    raw_search_results: list[dict]

    # research_summary:
    # - Writes: researcher_node (via summarizer tool)
    # - Reads: writer_node, critic_node, reviser_node
    # - Why: Supplies top 5-7 verified facts/articles ({title, summary, source_url, published_date}) to ground the draft without token context bloat.
    research_summary: list[dict]

    # draft_markdown:
    # - Writes: writer_node, reviser_node
    # - Reads: critic_node, human_review_node, publisher_node
    # - Why: Stores the active draft of the newsletter across iterative writing and revision cycles.
    draft_markdown: str

    # subject_line:
    # - Writes: writer_node, reviser_node
    # - Reads: publisher_node, app.py (frontend display), main.py (CLI display)
    # - Why: Provides an attention-grabbing, relevant email subject line generated alongside the body.
    subject_line: str

    # critique:
    # - Writes: critic_node
    # - Reads: should_continue_revision (conditional router), reviser_node, human_review_node
    # - Why: Carries structured evaluation {"pass": bool, "issues": list[str], "suggestions": str} driving the revision loop.
    critique: dict

    # revision_count:
    # - Writes: initial_state (initialized to 0), reviser_node (increments)
    # - Reads: should_continue_revision (conditional router), app.py, main.py
    # - Why: Tracks current loop count to prevent infinite critique-revision cycling.
    revision_count: int

    # max_revisions:
    # - Writes: User / initial_state
    # - Reads: should_continue_revision (conditional router)
    # - Why: Sets the hard ceiling on allowable revision iterations before forcing completion.
    max_revisions: int

    # human_feedback:
    # - Writes: human_review_node (populated via user input in HITL mode, empty in autonomous)
    # - Reads: reviser_node, publisher_node
    # - Why: Conveys human editorial guidance and specific rewrite instructions to the reviser.
    human_feedback: str

    # approved:
    # - Writes: critic_node (autonomous mode when critique passes), human_review_node (in HITL mode)
    # - Reads: should_continue_revision (conditional router), publisher_node, app.py
    # - Why: Acts as a safety gate to ensure unvetted or unapproved drafts are never published.
    approved: bool

    # final_html:
    # - Writes: publisher_node
    # - Reads: app.py (rendered email preview), main.py (CLI exporter)
    # - Why: Stores the finalized, responsive HTML email template ready for distribution.
    final_html: str

    # final_markdown:
    # - Writes: publisher_node
    # - Reads: app.py (markdown preview/copy), main.py (CLI exporter)
    # - Why: Stores clean, formatted markdown with verified citations for documentation/blog usage.
    final_markdown: str

    # output_path:
    # - Writes: publisher_node
    # - Reads: app.py, main.py
    # - Why: Records the filesystem path where published output files (.html and .md) were written.
    output_path: str

    # research_attempts:
    # - Writes: researcher_node (increments on each search cycle)
    # - Reads: should_continue_research (conditional router), researcher_node, UI
    # - Why: Tracks how many times the researcher node has attempted news retrieval to support autonomous retry.
    research_attempts: int

    # max_research_attempts:
    # - Writes: initial_state / user config
    # - Reads: should_continue_research (conditional router), researcher_node
    # - Why: Hard ceiling on research retry iterations (default: 3) before declaring research failure.
    max_research_attempts: int

    # research_failed:
    # - Writes: researcher_node (set to True if max_research_attempts exhausted with 0 summaries)
    # - Reads: should_continue_research, writer_node, critic_node
    # - Why: Signals downstream nodes to output an honest failure notice rather than hallucinating content.
    research_failed: bool

    # stage_log:
    # - Writes: planner_node, researcher_node, writer_node, critic_node, reviser_node, human_review_node, publisher_node
    # - Reads: app.py (real-time progress stepper), main.py (CLI log output)
    # - Why: Provides an auditable log of status messages (e.g. "Planning complete", "Critique: FAIL - reasons...") for the UI.
    stage_log: list[str]


def initial_state(
    goal: str,
    mode: Literal["autonomous", "human_in_loop"] = "autonomous",
    max_revisions: int = 2,
    max_research_attempts: int = 3
) -> NewsletterState:
    """
    Constructs a properly initialized starting state for the Newsletter Agent.

    Args:
        goal: The user's target topic, brief, or directive for the newsletter.
        mode: Execution mode, either 'autonomous' or 'human_in_loop'.
        max_revisions: Maximum allowable critique-revision iterations (default: 2).
        max_research_attempts: Maximum allowable research retry attempts (default: 3).

    Returns:
        NewsletterState: Fully initialized state ready for graph execution.
    """
    normalized_mode: Literal["autonomous", "human_in_loop"] = (
        "human_in_loop" if mode == "human_in_loop" else "autonomous"
    )

    return {
        "goal": goal.strip(),
        "mode": normalized_mode,
        "plan": "",
        "search_queries": [],
        "raw_search_results": [],
        "research_summary": [],
        "research_attempts": 0,
        "max_research_attempts": max_research_attempts,
        "research_failed": False,
        "draft_markdown": "",
        "subject_line": "",
        "critique": {
            "pass": False,
            "issues": [],
            "suggestions": ""
        },
        "revision_count": 0,
        "max_revisions": max_revisions,
        "human_feedback": "",
        "approved": False,
        "final_html": "",
        "final_markdown": "",
        "output_path": "",
        "stage_log": ["Agent initialized"],
    }
