"""
app.py: Streamlit Web Interface for the Newsletter Agent.

This module provides an interactive web application to run, monitor, and review
the LangGraph Newsletter Agent in both Fully Autonomous and Human-in-the-Loop (HITL) modes.

Architecture & State Lifecycle:
--------------------------------
1. Streamlit uses a rerun execution model: Every widget interaction reruns this script from top to bottom.
2. `st.session_state` preserves workflow state across reruns:
   - `thread_id`: Unique identifier tracking the agent conversation in the MemorySaver checkpointer.
   - `status`: Current execution state ("idle", "paused_for_review", "complete", "error").
   - `agent_response`: The latest response payload returned by run_newsletter_agent() or resume_newsletter_agent().
   - `request_changes_active`: Boolean flag toggling the feedback input form when a reviewer requests edits.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from config import get_secret, validate_gemini_key, validate_tavily_key
from agent.graph import run_newsletter_agent, resume_newsletter_agent, stream_newsletter_agent
from tools.html_generator import markdown_to_email_html

@st.cache_data(ttl=60, show_spinner=False)
def check_environment_status():
    """Validates Gemini and Tavily keys live using minimal test calls, cached for 60 seconds."""
    gem_ok, gem_msg = validate_gemini_key()
    tav_ok, tav_msg = validate_tavily_key()
    return gem_ok, gem_msg, tav_ok, tav_msg

# Load environment configuration
load_dotenv()

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="Newsletter Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
# Streamlit reruns the script on each button click. We initialize persistent keys
# so that the active session, thread_id, and agent deliverables survive page refreshes.

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

if "status" not in st.session_state:
    st.session_state["status"] = "idle"  # "idle" | "paused_for_review" | "complete" | "error"

if "agent_response" not in st.session_state:
    st.session_state["agent_response"] = None

if "request_changes_active" not in st.session_state:
    st.session_state["request_changes_active"] = False

if "error_message" not in st.session_state:
    st.session_state["error_message"] = None


# ==============================================================================
# Helper Renderers
# ==============================================================================

def render_timeline(stage_log: list[str]):
    """Renders stage_log entries as a stylized step-by-step audit trail."""
    if not stage_log:
        return

    st.markdown("### 📋 Agent Reasoning & Pipeline Timeline")
    with st.container(border=True):
        for idx, entry in enumerate(stage_log, 1):
            if "WARNING" in entry or "FAIL" in entry:
                st.markdown(f"⚠️ **Step {idx}:** {entry}")
            elif (
                "complete" in entry.lower()
                or "approved" in entry.lower()
                or "published" in entry.lower()
                or "pass" in entry.lower()
            ):
                st.markdown(f"✅ **Step {idx}:** {entry}")
            else:
                st.markdown(f"ℹ️ **Step {idx}:** {entry}")


def render_critique_card(critique: dict, revision_count: int):
    """Renders structured critique details with score and issues list."""
    if not critique:
        return

    passed = bool(critique.get("passed", False) or critique.get("pass", False))
    score = critique.get("score", "N/A")
    issues = critique.get("issues", [])
    suggestions = critique.get("suggestions", "")

    header_status = "✅ PASS" if passed else "❌ REVISION REQUIRED"
    with st.expander(f"🔍 Critique Details (Iteration {revision_count}) — {header_status} (Score: {score}/10)", expanded=True):
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Quality Score", f"{score}/10")
        m_col2.metric("Editorial Verdict", "Passed" if passed else "Needs Revision")

        if issues:
            st.markdown("**Identified Issues to Remediate:**")
            for issue in issues:
                st.markdown(f"- ⚠️ {issue}")
        else:
            st.markdown("✅ *No factual or structural issues detected.*")

        if suggestions:
            st.markdown(f"**Critic Suggestions:**\n> {suggestions}")


# ==============================================================================
# Header & Sidebar Configuration
# ==============================================================================

st.title("📰 Newsletter Agent")
st.caption(
    "Autonomous multi-step AI agent orchestrating research, drafting, rigorous "
    "self-critique, and responsive HTML publishing with Human-in-the-Loop review."
)
st.divider()

# Sidebar: Controls & API Status
with st.sidebar:
    st.header("⚙️ Configuration")

    mode_selection = st.radio(
        "Execution Mode:",
        options=["Fully Autonomous", "Human-in-the-Loop"],
        index=0,
        help="In Fully Autonomous mode, the agent plans, researches, drafts, self-critiques, and publishes automatically. "
             "In Human-in-the-Loop mode, the agent pauses after critique for your explicit editorial review and feedback."
    )
    mapped_mode = "autonomous" if mode_selection == "Fully Autonomous" else "human_in_loop"

    st.markdown("---")
    st.subheader("🔑 Environment Status")

    gem_ok, gem_msg, tav_ok, tav_msg = check_environment_status()

    # Gemini Status Display
    if gem_ok:
        st.markdown("✅ **Gemini API:** Ready")
        st.caption(f"_{gem_msg}_")
    else:
        st.markdown("❌ **Gemini API:** Missing/Invalid")
        st.caption(f"_{gem_msg}_")
        st.markdown("[👉 Get Gemini Key](https://aistudio.google.com/apikey)")

    # Tavily Status Display
    if tav_ok:
        st.markdown("✅ **Tavily API:** Ready")
        st.caption(f"_{tav_msg}_")
    else:
        st.markdown("❌ **Tavily API:** Missing/Invalid")
        st.caption(f"_{tav_msg}_")
        st.markdown("[👉 Get Tavily Key](https://tavily.com)")

    all_keys_valid = gem_ok and tav_ok

    if not all_keys_valid:
        st.warning("⚠️ Setup incomplete: Configure keys in `.env` and verify with `python scripts/check_setup.py`.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Re-test Keys", help="Clears cache and re-validates credentials"):
            check_environment_status.clear()
            st.rerun()
    with col_btn2:
        if st.button("🧹 Reset App", help="Clears session state to run a fresh newsletter"):
            st.session_state["thread_id"] = None
            st.session_state["status"] = "idle"
            st.session_state["agent_response"] = None
            st.session_state["request_changes_active"] = False
            st.session_state["error_message"] = None
            st.rerun()


# ==============================================================================
# Goal Input & Run Trigger
# ==============================================================================

goal_input = st.text_area(
    "Target Newsletter Goal:",
    value="Create a weekly newsletter on latest AI agent news and send it to our subscribers.",
    height=100,
    help="Define the topic, target angle, or audience for the newsletter."
)

if not all_keys_valid:
    st.error("🚨 API Keys Missing or Invalid: Please configure your `.env` file before running the agent. Check the sidebar for details.")

run_clicked = st.button("🚀 Run Agent", type="primary", use_container_width=True, disabled=not all_keys_valid)

# Handle Trigger
if run_clicked:
    if not all_keys_valid:
        st.error("Cannot run agent without valid API keys. Please configure .env first.")
    else:
        st.session_state["status"] = "running"
        st.session_state["request_changes_active"] = False
        st.session_state["error_message"] = None

        with st.status("🚀 Agent Workflow in Progress...", expanded=True) as status_box:
            try:
                final_response = None
                for node_name, update, data in stream_newsletter_agent(goal=goal_input.strip(), mode=mapped_mode):
                    if node_name == "planner":
                        queries = update.get("search_queries", [])
                        status_box.write(f"🧠 **1. Editorial Strategy:** Generated {len(queries)} targeted search queries.")
                    elif node_name == "researcher":
                        r_count = len(update.get("raw_search_results", []))
                        s_count = len(update.get("research_summary", []))
                        status_box.write(f"🔍 **2. News Retrieval (Tavily):** Fetched {r_count} web articles; synthesized {s_count} grounded stories.")
                    elif node_name == "writer":
                        subj = update.get("subject_line", "")
                        status_box.write(f"✍️ **3. Article Drafting:** Composed full newsletter draft ('{subj[:45]}...').")
                    elif node_name == "critic":
                        c = update.get("critique", {})
                        passed = c.get("passed", False)
                        score = c.get("score", "N/A")
                        verdict = "Passed ✅" if passed else "Revision Needed ⚠️"
                        status_box.write(f"🧐 **4. Editorial Critique:** Fact check score {score}/10 — {verdict}")
                    elif node_name == "reviser":
                        rev = update.get("revision_count", 1)
                        status_box.write(f"🔄 **Revision Loop:** Applying corrective feedback (Iteration {rev})...")
                    elif node_name == "publisher":
                        status_box.write("📦 **5. Publishing:** Inlined CSS styles, compiled HTML email, and saved deliverables.")
                    elif node_name in ["complete", "interrupt"]:
                        final_response = data

                if final_response:
                    st.session_state["thread_id"] = final_response.get("thread_id")
                    st.session_state["status"] = final_response.get("status")
                    st.session_state["agent_response"] = final_response
                    status_box.update(label="🎉 Pipeline Completed Successfully!", state="complete", expanded=False)
            except Exception as exc:
                st.session_state["status"] = "error"
                st.session_state["error_message"] = str(exc)
                status_box.update(label="❌ Pipeline Execution Failed", state="error", expanded=True)

        st.rerun()


# ==============================================================================
# Error Display
# ==============================================================================

if st.session_state["status"] == "error":
    st.error(f"❌ An error occurred during agent execution: {st.session_state['error_message']}")


# ==============================================================================
# Human-in-the-Loop (HITL) Review Panel
# ==============================================================================

if st.session_state["status"] == "paused_for_review":
    response = st.session_state.get("agent_response", {})
    payload = response.get("payload", {})
    thread_id = st.session_state.get("thread_id")

    draft_md = payload.get("draft_markdown", "")
    subject = payload.get("subject_line", "The Agentic Dispatch")
    critique = payload.get("critique", {})
    revision_count = payload.get("revision_count", 0)

    # Compile preview HTML for live email rendering
    preview_html = markdown_to_email_html(draft_md, subject)

    st.warning("⏸️ **Human Review Required:** The agent has paused graph execution for editorial review.")

    # Show Critique Card
    render_critique_card(critique, revision_count)

    # Side-by-side Draft Preview
    st.subheader(f"Subject Line: {subject}")
    preview_tabs = st.tabs(["📧 Rendered Email Preview", "📝 Raw Markdown Draft"])

    with preview_tabs[0]:
        components.html(preview_html, height=550, scrolling=True)

    with preview_tabs[1]:
        st.markdown(draft_md)

    st.markdown("---")
    st.markdown("### Editorial Decision")
    col_approve, col_reject = st.columns([1, 1])

    # Action 1: Approve & Publish
    with col_approve:
        if st.button("✅ Approve & Publish", type="primary", use_container_width=True):
            with st.spinner("Resuming agent with approval... publishing final newsletter..."):
                try:
                    resume_res = resume_newsletter_agent(thread_id, {"decision": "approve"})
                    st.session_state["status"] = resume_res.get("status")
                    st.session_state["agent_response"] = resume_res
                    st.session_state["request_changes_active"] = False
                except Exception as exc:
                    st.session_state["status"] = "error"
                    st.session_state["error_message"] = str(exc)
            st.rerun()

    # Action 2: Request Changes Toggle
    with col_reject:
        if st.button("🔁 Request Changes", use_container_width=True):
            st.session_state["request_changes_active"] = True
            st.rerun()

    # Feedback Input Form (when Request Changes is active)
    if st.session_state.get("request_changes_active"):
        with st.form("feedback_form"):
            feedback_text = st.text_area(
                "Editorial Feedback / Specific Changes Needed:",
                placeholder="E.g., Please add more emphasis on open-source frameworks and verify the enterprise benchmark figures...",
                height=120
            )
            submit_feedback = st.form_submit_button("📤 Submit Feedback & Revise Draft", type="primary")

            if submit_feedback:
                if not feedback_text.strip():
                    st.warning("Please enter feedback notes before submitting.")
                else:
                    with st.spinner("Resuming agent with feedback directives... revising draft..."):
                        try:
                            resume_res = resume_newsletter_agent(
                                thread_id,
                                {"decision": "request_changes", "feedback": feedback_text.strip()}
                            )
                            st.session_state["status"] = resume_res.get("status")
                            st.session_state["agent_response"] = resume_res
                            st.session_state["request_changes_active"] = False
                        except Exception as exc:
                            st.session_state["status"] = "error"
                            st.session_state["error_message"] = str(exc)
                    st.rerun()


# ==============================================================================
# Complete Deliverables Display
# ==============================================================================

if st.session_state["status"] == "complete":
    response = st.session_state.get("agent_response", {})
    result = response.get("result", {})

    final_html = result.get("final_html", "")
    final_md = result.get("final_markdown", "")
    output_path = result.get("output_path", "")
    subject_line = result.get("subject_line", "The Agentic Dispatch")
    revision_count = result.get("revision_count", 0)
    critique = result.get("critique", {})
    critique_score = critique.get("score", "N/A")
    stage_log = result.get("stage_log", [])

    st.success("🎉 **Newsletter Successfully Published!**")

    # Metrics Summary
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Status", "Published")
    metric_col2.metric("Total Revisions", revision_count)
    metric_col3.metric("Final Critique Score", f"{critique_score}/10")
    metric_col4.metric("Approved", "Yes")

    st.markdown(f"### 📬 Subject: **{subject_line}**")

    # Render Deliverables Tabs
    deliverable_tabs = st.tabs(["📧 Rendered Email Output", "📝 Markdown Content", "📁 Saved Artifacts"])

    with deliverable_tabs[0]:
        components.html(final_html, height=600, scrolling=True)

    with deliverable_tabs[1]:
        st.markdown(final_md)

    with deliverable_tabs[2]:
        st.markdown(f"**Local File Path:** `{output_path}`")
        if os.path.exists(output_path):
            st.info(f"File verified on disk ({os.path.getsize(output_path):,} bytes).")

    # Download Buttons
    st.markdown("---")
    st.subheader("📥 Export Deliverables")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.download_button(
            label="📄 Download HTML Email (.html)",
            data=final_html,
            file_name=os.path.basename(output_path) if output_path else "newsletter.html",
            mime="text/html",
            use_container_width=True,
        )

    with dl_col2:
        st.download_button(
            label="📝 Download Markdown Draft (.md)",
            data=final_md,
            file_name="newsletter.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Display Stage Log Timeline & Critique Card
    st.markdown("---")
    render_critique_card(critique, revision_count)
    render_timeline(stage_log)
