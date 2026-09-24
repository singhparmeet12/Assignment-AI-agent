"""
agent/nodes.py: Execution Nodes for the Newsletter Agent.

This module implements the complete 7-node LangGraph execution pipeline:
1. planner_node: Goal decomposition into editorial plan and targeted queries.
2. researcher_node: Grounded retrieval via Tavily and summarization via Gemini.
3. writer_node: Narrative synthesis into markdown draft with revision capability.
4. critic_node: Rigorous editorial evaluation against facts, coverage, and writing rubric.
5. reviser_node: Revision bookkeeping and transition counter.
6. human_review_node: Interactive checkpoint using interrupt() for HITL editorial approval.
7. publisher_node: Markdown-to-email HTML compilation, disk export, and simulated distribution.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Ensure workspace root is in sys.path when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt

from agent.state import NewsletterState
from config import get_secret
from tools.search import search_news
from tools.summarizer import synthesize_research
from tools.html_generator import markdown_to_email_html

# Load environment variables
load_dotenv()

# Configure module logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Prompt Constants
# ==============================================================================

PLANNER_SYSTEM_PROMPT = """You are the Lead Editorial Strategist for a premier AI technology newsletter.
Your mission is to analyze the user's objective and formulate a structured research plan along with high-precision search queries.

Guidelines:
1. Editorial Plan:
   - Outline the overarching narrative angle, target themes, and structure for the newsletter.
   - Define 3-4 thematic pillars to explore (e.g., framework launches, autonomous workflows, real-world deployment benchmarks, venture funding).
2. Search Queries:
   - Generate 3 to 5 targeted search queries.
   - Queries MUST be specific enough to surface recent, credible news and technical breakthroughs (e.g., 'AI agent frameworks launch 2025', 'autonomous agent enterprise deployment benchmark', 'multi-agent system startup funding').
   - Avoid generic terms like 'AI news' or 'machine learning updates'.
"""

PLANNER_USER_PROMPT = """Target Newsletter Goal:
{goal}

Generate the strategic research plan and 3-5 targeted search queries."""

WRITER_SYSTEM_PROMPT = """You are the Senior Editor & Staff Writer for 'The Agentic Dispatch', a publication focused on practical advancements in artificial intelligence agents and multi-agent systems.

Your objective is to transform curated research summaries and the editorial plan into an engaging, authoritative, publication-grade markdown newsletter.

Drafting Standards:
1. Subject Line:
   - Create a high-converting, curiosity-driven, and clear email subject line reflecting the key story.
2. Structure & Content:
   - Opening Hook: 1-2 paragraphs framing the state of the industry this week, connecting to the overall theme in the editorial plan.
   - Core Story Sections: For each article in the research summary:
     * H2 or H3 headline that communicates the key takeaway (not just the product name).
     * 2-4 sentences explaining what happened, why it matters, and technical or strategic takeaways.
     * Clean markdown link: [Read Source](URL) with the exact canonical URL provided in the research summary.
   - Synthesis Section: 1 paragraph highlighting the broader pattern uniting these developments.
   - Closing & Sign-off: An engaging sign-off encouraging reader thoughts.
3. Editorial Tone:
   - Sharp, analytical, accessible, and free of shallow buzzwords or filler hype.
"""

WRITER_REVISION_SECTION = """IMPORTANT REVISION INSTRUCTIONS:
This draft is a REVISION (Iteration {revision_count}).
You must explicitly address the following critique issues and suggestions rather than rewriting from scratch:

Critique Issues to Fix:
{critique_issues}

Critic Suggestions:
{critique_suggestions}

{human_feedback_text}

Preserve all accurate, strong portions of the existing draft while directly resolving every issue listed above."""

WRITER_USER_PROMPT = """Editorial Plan:
{plan}

Curated Research Summary:
{research_summary}

{existing_draft_context}

{revision_directives}

Please generate the complete markdown newsletter and subject line."""

CRITIC_SYSTEM_PROMPT = """You are the Senior Editorial Quality Director and Fact-Checking Chief for 'The Agentic Dispatch'.
Your job is to conduct a rigorous, objective editorial critique of newsletter drafts. You are NOT a rubber stamp, but you must be rational, factually grounded, and consistent with the provided research material.

Evaluation Rubric:
1. Factual Grounding:
   - Every factual claim, metric, and event in the draft MUST trace back directly to the provided research summary.
   - Flag any claims that appear hallucinated, embellished, or unsupported.
   - STRICT GROUNDING RULE: Never instruct the writer to add stories that do not exist in the research summary! Hallucinating ungrounded stories to artificially inflate story count is strictly forbidden.

2. Story Coverage & Quantity Criterion:
   - 5, 6, or 7 stories ALL FULLY PASS the coverage criterion. 5 stories is the valid floor, NEVER a failure!
   - Ground Truth Research Availability: The writer had N summaries available in the research summary (specified in context).
   - When to PASS coverage:
     * If the draft covers 5, 6, or 7 distinct stories, coverage PASSES.
     * If the draft covers all N available research summaries (e.g. N = 5), coverage PASSES.
   - When to FLAG coverage:
     * ONLY flag 'insufficient coverage' if M < N (the writer had N summaries available but omitted some, covering M < N stories without justification).
     * OR if M < 5 (below the absolute floor of 5 stories), UNLESS N itself was fewer than 5.
   - PROHIBITED CRITIQUE: NEVER fail a draft or demand an additional story simply because it has 5 stories! If the draft covers 5, 6, or 7 stories, or if M == N, coverage MUST be approved.

3. Writing Quality & Engagement:
   - The prose must be engaging, analytical, and fluid—not dry, robotic, or a mere bulleted list of links.
   - It should clearly explain *why* each development matters to practitioners.

4. Structural Integrity:
   - Ensure the presence of: a clear opening hook/intro, well-demarcated sections with headline + summary, verified source links [Read Source](URL), a macro trend synthesis, and an engaging sign-off.

5. Revision Verification (if revision_count > 0):
   - Review previous critique issues and verify whether the author genuinely corrected them.

Decision Threshold:
- Set 'passed' to TRUE if the draft meets all standards with no significant factual, structural, or quality flaws (typically score 8-10).
- If the draft has genuine factual hallucinations, omitted available research summaries (M < N or M < 5), missing links, or robotic prose, set 'passed' to FALSE, specify the exact issues as concrete bullet points, and provide clear, actionable suggestions.
- Remember: 5 stories is a complete pass for coverage if all available research was used.
"""

CRITIC_USER_PROMPT = """Editorial Ground Truth & Research Context:
- Available Research Summaries provided to Writer: {available_count} summaries (N = {available_count})
- Coverage Rule: 5, 6, or 7 stories all PASS. Only flag coverage if draft used M < {available_count} available summaries or M < 5.

Curated Research Summary (Ground Truth):
{research_summary}

Draft Markdown to Evaluate:
{draft_markdown}

Subject Line:
{subject_line}

Revision Context:
- Current Revision Count: {revision_count}
{previous_critique_context}

Perform your rigorous evaluation according to the rubric."""


# ==============================================================================
# Shared LLM Client Setup & Credential Verification
# ==============================================================================

def verify_gemini_credentials() -> str:
    """
    Verifies that a valid Google Gemini API key is configured.
    Raises ValueError with a clear, loud error message if the key is missing or is a placeholder.
    """
    api_key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_") or api_key == "dummy_key_for_offline_validation":
        error_msg = (
            "\n" + "=" * 74 + "\n"
            "🚨 CRITICAL CONFIGURATION ERROR: GOOGLE_API_KEY (or GEMINI_API_KEY) is missing or invalid!\n"
            "   The Newsletter Agent requires a valid Google Gemini API key for planning, writing, and critique.\n"
            "   Proceeding without a valid key will cause failed generation or hollow newsletters.\n\n"
            "   👉 Action Required:\n"
            "      1. Get a free API key at: https://aistudio.google.com/apikey\n"
            "      2. Set it in your .env file: GOOGLE_API_KEY=your_actual_key_here\n"
            "      3. Verify your configuration: python scripts/check_setup.py\n"
            + "=" * 74
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    return api_key


def get_shared_gemini_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Returns an initialized ChatGoogleGenerativeAI client configured with a specific temperature.
    Validates API key and ensures fresh credentials from get_secret().
    
    Args:
        temperature: Sampling temperature (lower for factual/planning, higher for writing).
    
    Returns:
        ChatGoogleGenerativeAI: Initialized Gemini model instance.
    """
    api_key = verify_gemini_credentials()
    model_name = get_secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        max_retries=2,
        timeout=120,
    )


# Module-level instances initialized lazily when nodes run
planner_llm = None
researcher_llm = None
writer_llm = None
critic_llm = None


# ==============================================================================
# Pydantic Schemas for Structured Node Outputs
# ==============================================================================

class PlannerOutput(BaseModel):
    """Structured output schema for the planner_node."""
    plan: str = Field(
        ...,
        description="Comprehensive editorial research plan and narrative angle for the newsletter."
    )
    search_queries: list[str] = Field(
        ...,
        description="List of 3 to 5 targeted, high-precision search queries."
    )


class WriterOutput(BaseModel):
    """Structured output schema for the writer_node."""
    subject_line: str = Field(
        ...,
        description="Compelling, high-open-rate subject line for the email newsletter."
    )
    draft_markdown: str = Field(
        ...,
        description="Complete newsletter draft written in clean, well-formatted markdown."
    )


class CritiqueResult(BaseModel):
    """Structured output schema for the critic_node."""
    passed: bool = Field(
        ...,
        description="True ONLY if the draft satisfies all editorial standards and factual grounding criteria; False if issues need remediation."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific, actionable problems found in the draft (empty if passed)."
    )
    suggestions: str = Field(
        ...,
        description="Concrete guidance and instructions for the reviser to fix the identified problems."
    )
    score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Overall quality score from 1 to 10 evaluating grounding, coverage, and writing."
    )


# ==============================================================================
# Node 1: Planner Node
# ==============================================================================

def planner_node(state: NewsletterState) -> dict:
    """
    LangGraph Planner Node:
    Takes the user goal, decomposes it into an editorial narrative structure,
    and produces 3-5 specific, timely search queries to retrieve relevant AI news.

    Args:
        state: Current NewsletterState.

    Returns:
        dict: State update containing 'plan', 'search_queries', and updated 'stage_log'.
    """
    t0 = time.perf_counter()
    goal = state.get("goal", "").strip() or "Create a weekly newsletter covering the latest AI agent breakthroughs"
    logger.info(f"Running planner_node for goal: '{goal[:80]}...'")

    prompt_messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": PLANNER_USER_PROMPT.format(goal=goal)},
    ]

    plan_text = ""
    queries: list[str] = []

    # Initialize validated Gemini client
    planner_llm = get_shared_gemini_llm(temperature=0.2)

    # Primary path: Structured output via Gemini
    try:
        if hasattr(planner_llm, "with_structured_output"):
            structured_planner = planner_llm.with_structured_output(PlannerOutput)
            result = structured_planner.invoke(prompt_messages)
            if isinstance(result, PlannerOutput):
                plan_text = result.plan.strip()
                queries = [q.strip() for q in result.search_queries if q.strip()]
            elif isinstance(result, dict):
                plan_text = result.get("plan", "").strip()
                queries = [q.strip() for q in result.get("search_queries", []) if q.strip()]

    except Exception as err:
        logger.warning(f"Structured planner output failed: {err}. Attempting fallback invocation...")

    # Secondary path: Standard invocation with JSON parsing
    if not plan_text or not queries:
        try:
            fallback_prompt = (
                f"{PLANNER_SYSTEM_PROMPT}\n\n"
                "Return a valid JSON object matching this schema:\n"
                "{\n"
                '  "plan": "Editorial narrative and outline...",\n'
                '  "search_queries": ["query 1", "query 2", "query 3"]\n'
                "}\n\n"
                f"{PLANNER_USER_PROMPT.format(goal=goal)}"
            )
            response = planner_llm.invoke(fallback_prompt)
            content = response.content if hasattr(response, "content") else str(response)

            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                data = json.loads(content[start_idx : end_idx + 1])
                plan_text = data.get("plan", "").strip()
                queries = [q.strip() for q in data.get("search_queries", []) if q.strip()]
        except Exception as fallback_err:
            logger.error(f"Fallback planner parsing failed: {fallback_err}")

    # Safety default if LLM failed or API key was absent
    if not plan_text:
        plan_text = (
            f"Editorial Plan for '{goal}':\n"
            "1. Industry Overview: Key autonomous agent breakthroughs this week.\n"
            "2. Tooling & Frameworks: Open-source agent releases and benchmarks.\n"
            "3. Strategic Impact: Enterprise adoption and future outlook."
        )
    if not queries:
        queries = [
            f"{goal} breakthroughs 2025",
            "autonomous AI agent framework release news",
            "multi-agent systems enterprise adoption",
        ]

    duration = round(time.perf_counter() - t0, 2)
    stage_entry = f"Planning complete: {len(queries)} queries generated [{duration:.1f}s]"
    logger.info(stage_entry)

    current_log = state.get("stage_log", [])
    prev_timings = dict(state.get("stage_timings", {}))
    return {
        "plan": plan_text,
        "search_queries": queries,
        "stage_log": current_log + [stage_entry],
        "stage_timings": {**prev_timings, "planning": duration},
    }


# ==============================================================================
# Node 2: Researcher Node
# ==============================================================================

def researcher_node(state: NewsletterState) -> dict:
    t0 = time.perf_counter()
    attempt = state.get("research_attempts", 0) + 1
    max_attempts = state.get("max_research_attempts", 3)
    original_queries = state.get("search_queries", [])
    goal = state.get("goal", "AI agents")
    current_log = state.get("stage_log", [])

    # Configure progressive broadening based on attempt number
    active_queries = list(original_queries)
    if attempt == 1:
        days_back = 30
        attempt_msg = f"Research attempt 1/{max_attempts}: searching with 30-day window"
    elif attempt == 2:
        days_back = 90
        attempt_msg = f"Research attempt 2/{max_attempts}: broadening search window to 90 days"
    else:
        days_back = 180
        # Simplify queries to broader, less jargon-heavy keywords
        simplified_queries = [
            f"{goal} news",
            "autonomous AI agents developments",
            "AI agent framework launches",
            "generative AI agents breakthroughs",
        ]
        active_queries = simplified_queries
        attempt_msg = (
            f"Research attempt {attempt}/{max_attempts}: simplifying queries to "
            f"['{goal} news', 'autonomous AI agents developments', ...], window 180 days"
        )

    logger.info(attempt_msg)

    # Step 1: Execute Tavily search with active parameters (instrument search timing)
    t_search_0 = time.perf_counter()
    raw_results = search_news(queries=active_queries, max_results_per_query=5, days_back=days_back)
    search_duration = round(time.perf_counter() - t_search_0, 2)

    # Step 2: Synthesize and rank findings with the shared Gemini LLM (instrument synthesis timing)
    t_synth_0 = time.perf_counter()
    researcher_llm = get_shared_gemini_llm(temperature=0.1)
    topic_hint = state.get("goal", "AI agents")
    synthesized_summaries = synthesize_research(
        raw_results=raw_results,
        llm=researcher_llm,
        top_n=7,
        focus_topic=topic_hint
    )
    synth_duration = round(time.perf_counter() - t_synth_0, 2)
    total_research_duration = round(time.perf_counter() - t0, 2)

    new_logs = [attempt_msg]
    timing_suffix = f"[{total_research_duration:.1f}s: {search_duration:.1f}s search + {synth_duration:.1f}s synthesis]"

    if synthesized_summaries:
        research_failed = False
        outcome_msg = (
            f"Research complete (attempt {attempt}/{max_attempts}): found {len(raw_results)} articles, "
            f"synthesized {len(synthesized_summaries)} summaries {timing_suffix}"
        )
        logger.info(outcome_msg)
        new_logs.append(outcome_msg)
    else:
        if attempt < max_attempts:
            research_failed = False
            outcome_msg = (
                f"Research attempt {attempt}/{max_attempts} returned 0 summaries {timing_suffix}. "
                "Triggering self-correction retry with broader parameters..."
            )
            logger.warning(outcome_msg)
            new_logs.append(outcome_msg)
        else:
            research_failed = True
            outcome_msg = f"Research failed after {max_attempts} attempts — no grounded sources found {timing_suffix}."
            logger.warning(outcome_msg)
            new_logs.append(outcome_msg)

    prev_timings = dict(state.get("stage_timings", {}))
    return {
        "research_attempts": attempt,
        "search_queries": active_queries,
        "raw_search_results": raw_results,
        "research_summary": synthesized_summaries,
        "research_failed": research_failed,
        "stage_log": current_log + new_logs,
        "stage_timings": {
            **prev_timings,
            "research": total_research_duration,
            "research_search": search_duration,
            "research_synthesis": synth_duration,
        },
    }


# ==============================================================================
# Node 3: Writer Node
# ==============================================================================

def writer_node(state: NewsletterState) -> dict:
    t0 = time.perf_counter()
    plan = state.get("plan", "Standard AI Agent Weekly Overview")
    research_summary = state.get("research_summary", [])
    critique = state.get("critique", {})
    human_feedback = state.get("human_feedback", "").strip()
    revision_count = state.get("revision_count", 0)
    existing_draft = state.get("draft_markdown", "").strip()

    logger.info(f"Running writer_node (revision: {revision_count})...")

    # If research failed or no verified summaries exist, output an honest factual notice
    # rather than hallucinating/fabricating a 'conceptual' newsletter.
    if state.get("research_failed") or not research_summary:
        goal = state.get("goal", "AI Agents")
        attempts = state.get("research_attempts", 3)
        queries_executed = state.get("search_queries", [])
        queries_fmt = ", ".join(f"`{q}`" for q in queries_executed) if queries_executed else "Standard topical queries"

        subject_line = f"The Agentic Dispatch: Editorial Notice — No Verified Sources Found ({goal})"
        draft_markdown = f"""# The Agentic Dispatch: Editorial Notice

## Automated Verification Status: No Verified Sources This Cycle

**Target Brief:** {goal}

During this editorial cycle, our autonomous research agent conducted **{attempts} iterative search attempts** across expanded timeframes (up to 180 days) and simplified topical queries. However, no authoritative, primary news articles meeting our verification and grounding standards were successfully retrieved.

### Our Commitment to Factuality
To uphold editorial integrity, *The Agentic Dispatch* does not fabricate placeholder narratives, simulate unsourced breakthroughs, or generate unverified stories. 

- **Queries Executed:** {queries_fmt}
- **Action Taken:** Automated publishing halted for unsourced material; research parameters will recalibrate on the next scheduled run.

We will resume standard weekly briefings as soon as verified developments are confirmed by primary sources.

*— The Agentic Dispatch Editorial Team*
"""
        stage_entry = f"Draft written: Honest research hiatus notice (no hallucinated content, attempt {attempts})"
        logger.info(stage_entry)
        return {
            "draft_markdown": draft_markdown,
            "subject_line": subject_line,
            "stage_log": state.get("stage_log", []) + [stage_entry],
        }

    # Format research items for prompt injection
    research_formatted_items = []
    for idx, item in enumerate(research_summary, 1):
        research_formatted_items.append(
            f"Story {idx}:\n"
            f"  Headline: {item.get('title', 'Untitled')}\n"
            f"  Summary: {item.get('summary', '')}\n"
            f"  Source URL: {item.get('source_url', '')}\n"
            f"  Published Date: {item.get('published_date', 'Recent')}"
        )
    research_text = "\n\n".join(research_formatted_items)

    # Check if this invocation is a revision
    is_revision = bool(
        (isinstance(critique, dict) and critique.get("issues"))
        or human_feedback
        or revision_count > 0
    )

    revision_directives = ""
    existing_draft_context = ""

    if is_revision and existing_draft:
        critique_issues_list = critique.get("issues", []) if isinstance(critique, dict) else []
        critique_issues_str = "\n".join(f"- {issue}" for issue in critique_issues_list) or "None noted."
        critique_suggestions_str = critique.get("suggestions", "Refine and improve flow.") if isinstance(critique, dict) else "Improve flow."
        
        human_text = f"Human Editor Notes:\n{human_feedback}" if human_feedback else "Human Editor Notes: None provided."

        revision_directives = WRITER_REVISION_SECTION.format(
            revision_count=revision_count,
            critique_issues=critique_issues_str,
            critique_suggestions=critique_suggestions_str,
            human_feedback_text=human_text,
        )
        existing_draft_context = f"Current Draft to Revise:\n\n{existing_draft}"

    # Build prompt payload
    user_prompt_content = WRITER_USER_PROMPT.format(
        plan=plan,
        research_summary=research_text,
        existing_draft_context=existing_draft_context,
        revision_directives=revision_directives,
    )

    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_content},
    ]

    subject_line = ""
    draft_markdown = ""

    # Initialize validated Gemini client
    writer_llm = get_shared_gemini_llm(temperature=0.7)

    # Primary path: Structured output via Gemini
    try:
        if hasattr(writer_llm, "with_structured_output"):
            structured_writer = writer_llm.with_structured_output(WriterOutput)
            result = structured_writer.invoke(messages)
            if isinstance(result, WriterOutput):
                subject_line = result.subject_line.strip()
                draft_markdown = result.draft_markdown.strip()
            elif isinstance(result, dict):
                subject_line = result.get("subject_line", "").strip()
                draft_markdown = result.get("draft_markdown", "").strip()

    except Exception as err:
        logger.warning(f"Structured writer output failed: {err}. Attempting fallback invocation...")

    # Secondary path: Standard invocation with parsing
    if not draft_markdown:
        try:
            fallback_prompt = (
                f"{WRITER_SYSTEM_PROMPT}\n\n"
                "Return a valid JSON object matching this schema:\n"
                "{\n"
                '  "subject_line": "Catchy Subject Line",\n'
                '  "draft_markdown": "# Title\\n\\nFull markdown content..."\n'
                "}\n\n"
                f"{user_prompt_content}"
            )
            response = writer_llm.invoke(fallback_prompt)
            content = response.content if hasattr(response, "content") else str(response)

            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                data = json.loads(content[start_idx : end_idx + 1])
                subject_line = data.get("subject_line", "").strip()
                draft_markdown = data.get("draft_markdown", "").strip()
            else:
                # Raw text fallback
                draft_markdown = content.strip()
                subject_line = "The Agentic Dispatch: Weekly Intelligence Briefing"
        except Exception as fallback_err:
            logger.error(f"Fallback writer generation failed: {fallback_err}")
            # Final offline fallback
            subject_line = "The Agentic Dispatch: Latest AI Agent Developments"
            draft_markdown = f"# {subject_line}\n\n## Overview\n{plan}\n\n## Stories\n{research_text}"

    if not subject_line:
        subject_line = "The Agentic Dispatch: Autonomous Intelligence Weekly"

    duration = round(time.perf_counter() - t0, 2)
    stage_entry = f"Draft written (revision {revision_count}) [{duration:.1f}s]"
    logger.info(stage_entry)

    current_log = state.get("stage_log", [])
    prev_timings = dict(state.get("stage_timings", {}))
    prev_writing = prev_timings.get("writing", [])
    if not isinstance(prev_writing, list):
        prev_writing = [prev_writing] if prev_writing else []
    new_writing = prev_writing + [duration]

    return {
        "draft_markdown": draft_markdown,
        "subject_line": subject_line,
        "stage_log": current_log + [stage_entry],
        "stage_timings": {**prev_timings, "writing": new_writing},
    }


# ==============================================================================
# The Self-Critique & Revision Loop (Nodes 4 & 5)
# ==============================================================================
"""
The Self-Critique & Revision Loop:
----------------------------------
1. critic_node evaluates the current draft against factual grounding, story coverage,
   structure, and prose quality.
2. If critique passes (or max_revisions is reached), the pipeline routes forward to
   human review (HITL mode) or direct publishing (autonomous mode).
3. If critique fails and revisions remain (revision_count < max_revisions):
   - reviser_node increments the revision counter and logs the attempt.
   - The graph routes back to writer_node.
   - writer_node receives the updated state containing state["critique"]["issues"] and
     state["critique"]["suggestions"] (plus any human feedback).
   - writer_node produces a revised draft specifically targeting those flaws.
   - The revised draft routes back to critic_node for re-evaluation.
4. Loop repeats until the draft passes critique or max_revisions is reached.
   This hard limit guarantees the pipeline will never get stuck in an infinite loop.
"""


def critic_node(state: NewsletterState) -> dict:
    t0 = time.perf_counter()
    draft_markdown = state.get("draft_markdown", "").strip()
    research_summary = state.get("research_summary", [])
    subject_line = state.get("subject_line", "Untitled")
    revision_count = state.get("revision_count", 0)
    previous_critique = state.get("critique", {})

    logger.info(f"Running critic_node (iteration {revision_count})...")

    # If research failed and an honest notice was drafted, bypass normal rubric to avoid infinite loops
    if state.get("research_failed") or not research_summary:
        stage_entry = "Critique: PASS (honest research hiatus notice accepted without fabrication)"
        logger.info(stage_entry)
        return {
            "critique": {
                "passed": True,
                "pass": True,
                "issues": [],
                "suggestions": "Honest failure notice verified. Factual integrity upheld without hallucinations.",
                "score": 10,
            },
            "stage_log": state.get("stage_log", []) + [stage_entry],
        }

    # Format research summaries as ground truth
    if research_summary:
        research_bullets = []
        for idx, item in enumerate(research_summary, 1):
            research_bullets.append(
                f"[{idx}] {item.get('title', 'Untitled')}\n"
                f"     Summary: {item.get('summary', '')}\n"
                f"     URL: {item.get('source_url', '')}"
            )
        research_corpus = "\n\n".join(research_bullets)
    else:
        research_corpus = "No research summaries available (empty research state)."

    # Format previous critique context if this is a subsequent revision
    if revision_count > 0 and isinstance(previous_critique, dict) and previous_critique.get("issues"):
        prev_issues = "\n".join(f"- {iss}" for iss in previous_critique.get("issues", []))
        previous_context = (
            f"- Previous Critique Issues:\n{prev_issues}\n"
            f"- Previous Suggestions: {previous_critique.get('suggestions', 'None')}"
        )
    else:
        previous_context = "- Initial draft review (no prior critique)."

    available_count = len(research_summary)
    user_prompt = CRITIC_USER_PROMPT.format(
        available_count=available_count,
        research_summary=research_corpus,
        draft_markdown=draft_markdown or "[Empty draft]",
        subject_line=subject_line,
        revision_count=revision_count,
        previous_critique_context=previous_context,
    )

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    passed = False
    issues: list[str] = []
    suggestions = ""
    score = 5

    # Initialize validated Gemini client
    critic_llm = get_shared_gemini_llm(temperature=0.2)

    # Primary path: Structured output via Gemini
    try:
        if hasattr(critic_llm, "with_structured_output"):
            structured_critic = critic_llm.with_structured_output(CritiqueResult)
            result = structured_critic.invoke(messages)
            if isinstance(result, CritiqueResult):
                passed = result.passed
                issues = [str(i).strip() for i in result.issues if str(i).strip()]
                suggestions = result.suggestions.strip()
                score = result.score
            elif isinstance(result, dict):
                passed = bool(result.get("passed", False))
                issues = [str(i).strip() for i in result.get("issues", []) if str(i).strip()]
                suggestions = result.get("suggestions", "").strip()
                score = int(result.get("score", 5))

    except Exception as err:
        logger.warning(f"Structured critic output failed: {err}. Attempting fallback invocation...")

    # Secondary path: Standard invocation with JSON parsing
    if not suggestions:
        try:
            fallback_prompt = (
                f"{CRITIC_SYSTEM_PROMPT}\n\n"
                "Return a valid JSON object matching this schema:\n"
                "{\n"
                '  "passed": false,\n'
                '  "issues": ["Issue 1", "Issue 2"],\n'
                '  "suggestions": "Concrete instructions...",\n'
                '  "score": 6\n'
                "}\n\n"
                f"{user_prompt}"
            )
            response = critic_llm.invoke(fallback_prompt)
            content = response.content if hasattr(response, "content") else str(response)

            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                data = json.loads(content[start_idx : end_idx + 1])
                passed = bool(data.get("passed", False))
                issues = [str(i).strip() for i in data.get("issues", []) if str(i).strip()]
                suggestions = data.get("suggestions", "").strip()
                score = int(data.get("score", 5))
        except Exception as fallback_err:
            logger.error(f"Fallback critic parsing failed: {fallback_err}")

    # Safety default for offline/test environments
    if not suggestions:
        has_links = "[" in draft_markdown and "]" in draft_markdown and "http" in draft_markdown
        has_sections = draft_markdown.count("##") >= 3 or draft_markdown.count("###") >= 3
        has_length = len(draft_markdown) >= 300

        if has_links and has_sections and has_length:
            passed = True
            issues = []
            suggestions = "Draft meets structural and sourcing standards."
            score = 8
        else:
            passed = False
            issues = []
            if not has_length:
                issues.append("Draft is too brief and lacks substantive narrative depth.")
            if not has_links:
                issues.append("Missing source hyperlink citations for one or more stories.")
            if not has_sections:
                issues.append("Story sections are insufficiently demarcated with H2/H3 headers.")
            suggestions = "Expand story sections with 2-4 analytical sentences and insert verified source URLs."
            score = 5

    # Programmatic Safeguard: Prevent contradictory coverage critiques
    # Determine the number of stories covered in the draft
    link_count = (
        draft_markdown.count("[Read Source]")
        + draft_markdown.count("[Read More]")
        + draft_markdown.count("[Source]")
    )
    if link_count == 0:
        import re
        link_count = len(set(re.findall(r'https?://[^\s\)]+', draft_markdown)))
    header_count = max(0, draft_markdown.count("## ") - 2)
    draft_story_count = max(link_count, header_count)

    if issues:
        filtered_issues = []
        for issue in issues:
            lower_issue = issue.lower()
            # Detect complaints about covering "only 5" stories or demanding additional stories
            is_coverage_complaint = (
                ("only 5" in lower_issue or "just 5" in lower_issue or "covers 5" in lower_issue or "5 distinct" in lower_issue)
                or ("additional distinct story" in lower_issue or "expand to include an additional" in lower_issue or "add an additional story" in lower_issue or "add another story" in lower_issue)
                or ("fewer stories" in lower_issue and (draft_story_count >= 5 or draft_story_count >= available_count))
                or ("insufficient coverage" in lower_issue and (draft_story_count >= 5 or draft_story_count >= available_count))
            )
            # If the draft covered all available research summaries or at least 5 stories, this critique is invalid
            if is_coverage_complaint and (draft_story_count >= 5 or draft_story_count >= available_count):
                logger.info(
                    f"Programmatic safeguard: filtered invalid coverage critique '{issue}' "
                    f"(draft covers {draft_story_count} stories with {available_count} summaries available)."
                )
            else:
                filtered_issues.append(issue)

        issues = filtered_issues

    # If the only reason it failed was the filtered coverage critique, mark as passed
    if not passed and not issues and suggestions:
        passed = True
        score = max(score, 8)
        suggestions = "Draft coverage satisfies the 5-7 story requirement and fully utilizes available research summaries."

    status_str = "PASS" if passed else "FAIL"
    duration = round(time.perf_counter() - t0, 2)
    stage_entry = (
        f"Critique (revision {revision_count}): {status_str} - score {score}/10, {len(issues)} issues [{duration:.1f}s]"
    )
    logger.info(stage_entry)

    current_log = state.get("stage_log", [])
    prev_timings = dict(state.get("stage_timings", {}))
    prev_critique = prev_timings.get("critique", [])
    if not isinstance(prev_critique, list):
        prev_critique = [prev_critique] if prev_critique else []
    new_critique = prev_critique + [duration]

    return {
        "critique": {
            "passed": passed,
            "pass": passed,
            "issues": issues,
            "suggestions": suggestions,
            "score": score,
        },
        "stage_log": current_log + [stage_entry],
        "stage_timings": {**prev_timings, "critique": new_critique},
    }


def reviser_node(state: NewsletterState) -> dict:
    t0 = time.perf_counter()
    next_revision = state.get("revision_count", 0) + 1
    max_revisions = state.get("max_revisions", 2)
    duration = round(time.perf_counter() - t0, 2)
    stage_entry = f"Revising draft (attempt {next_revision}/{max_revisions})... [{duration:.1f}s]"
    logger.info(stage_entry)

    current_log = state.get("stage_log", [])
    prev_timings = dict(state.get("stage_timings", {}))
    prev_revising = prev_timings.get("revising", [])
    if not isinstance(prev_revising, list):
        prev_revising = [prev_revising] if prev_revising else []
    new_revising = prev_revising + [duration]

    return {
        "revision_count": next_revision,
        "stage_log": current_log + [stage_entry],
        "stage_timings": {**prev_timings, "revising": new_revising},
    }


# ==============================================================================
# Nodes 6 & 7: Human Review & Publisher Nodes
# ==============================================================================

def human_review_node(state: NewsletterState) -> dict:
    t0 = time.perf_counter()
    mode_val = state.get("mode", "unknown")
    logger.info(f"human_review_node ENTERED - mode={mode_val}, about to call interrupt()")
    print(f"human_review_node ENTERED - mode={mode_val}, about to call interrupt()", flush=True)
    draft = state.get("draft_markdown", "")
    subject = state.get("subject_line", "The Agentic Dispatch")
    critique = state.get("critique", {})
    revision_count = state.get("revision_count", 0)

    interrupt_payload = {
        "draft_markdown": draft,
        "subject_line": subject,
        "critique": critique,
        "revision_count": revision_count,
        "instruction": "Please review the newsletter draft and decide whether to approve or request changes.",
    }

    logger.info(
        f"Entering human_review_node (iteration {revision_count}). "
        "Triggering graph interrupt for human editorial review..."
    )

    t_prep = time.perf_counter() - t0

    # Genuine LangGraph interruption: execution suspends here until resumed with input
    user_decision = interrupt(interrupt_payload)
    t_resume = time.perf_counter()
    logger.info(f"Human review resumed with decision payload: {user_decision}")

    # Robust parsing of human decision
    is_approved = False
    feedback_text = ""

    if isinstance(user_decision, dict):
        decision_val = str(user_decision.get("decision", "")).strip().lower()
        if (
            decision_val in ("approve", "approved", "yes", "pass", "ok")
            or user_decision.get("approved") is True
        ):
            is_approved = True
        else:
            is_approved = False
            feedback_text = str(
                user_decision.get("feedback")
                or user_decision.get("human_feedback")
                or user_decision.get("comments")
                or ""
            ).strip()
    elif isinstance(user_decision, str):
        if user_decision.strip().lower() in ("approve", "approved", "yes", "ok", "pass"):
            is_approved = True
        else:
            is_approved = False
            feedback_text = user_decision.strip()
    elif user_decision is True:
        is_approved = True

    review_duration = round(t_prep + (time.perf_counter() - t_resume), 2)
    current_log = state.get("stage_log", [])
    prev_timings = dict(state.get("stage_timings", {}))

    if is_approved:
        stage_entry = f"Human approved draft [{review_duration:.1f}s]"
        logger.info(stage_entry)
        return {
            "approved": True,
            "human_feedback": "",
            "stage_log": current_log + [stage_entry],
            "stage_timings": {**prev_timings, "human_review": review_duration},
        }
    else:
        stage_entry = (
            f"Human requested changes: {feedback_text} [{review_duration:.1f}s]"
            if feedback_text
            else f"Human requested changes without specific feedback notes [{review_duration:.1f}s]"
        )
        logger.info(stage_entry)
        return {
            "approved": False,
            "human_feedback": feedback_text,
            "stage_log": current_log + [stage_entry],
            "stage_timings": {**prev_timings, "human_review": review_duration},
        }


def publisher_node(state: NewsletterState) -> dict:
    t0 = time.perf_counter()
    draft_markdown = state.get("draft_markdown", "").strip()
    subject_line = state.get("subject_line", "The Agentic Dispatch").strip()
    logger.info(f"Running publisher_node for subject: '{subject_line}'...")

    # Step 1: Generate responsive email HTML
    final_html = markdown_to_email_html(
        markdown_text=draft_markdown,
        subject_line=subject_line,
        preheader="Curated weekly technical insights and breakthroughs in AI agents."
    )

    # Step 2: Ensure outputs/ directory exists and save deliverables
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"newsletter_{timestamp_str}.html"
    md_filename = f"newsletter_{timestamp_str}.md"

    html_path = outputs_dir / html_filename
    md_path = outputs_dir / md_filename

    html_path.write_text(final_html, encoding="utf-8")
    md_path.write_text(draft_markdown, encoding="utf-8")

    # Step 3: Simulated email transmission logging
    separator = "=" * 60
    preview_snippet = draft_markdown[:300].replace("\n", " ").strip()
    send_banner = (
        f"\n{separator}\n"
        "=== SIMULATED EMAIL SEND ===\n"
        f"To:          Subscribers of 'The Agentic Dispatch' (12,450 recipients)\n"
        f"Subject:     {subject_line}\n"
        f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"HTML File:   {html_path.as_posix()}\n"
        f"MD File:     {md_path.as_posix()}\n"
        f"Snippet:     {preview_snippet}...\n"
        f"{separator}\n"
    )
    print(send_banner)
    logger.info(f"Email simulation output logged. Saved to {html_path.as_posix()}")

    duration = round(time.perf_counter() - t0, 2)
    prev_timings = dict(state.get("stage_timings", {}))
    total_time = duration
    for k, v in prev_timings.items():
        if k in ["research_search", "research_synthesis", "total"]:
            continue
        if isinstance(v, (int, float)):
            total_time += v
        elif isinstance(v, list):
            total_time += sum(x for x in v if isinstance(x, (int, float)))
    total_time = round(total_time, 2)

    new_timings = {
        **prev_timings,
        "publish": duration,
        "total": total_time,
    }

    stage_entry = (
        f"Newsletter published: saved to {html_path.as_posix()}, simulated send complete "
        f"[Total pipeline time: {total_time:.1f}s]"
    )
    current_log = state.get("stage_log", [])

    return {
        "final_html": final_html,
        "final_markdown": draft_markdown,
        "output_path": str(html_path.as_posix()),
        "approved": True,
        "stage_log": current_log + [stage_entry],
        "stage_timings": new_timings,
    }
