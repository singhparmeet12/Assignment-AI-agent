"""
app.py: Streamlit Web Interface for the Newsletter Agent.

Provides an enterprise-grade AI Agent interface to configure, execute, monitor,
and review the LangGraph Newsletter Agent in both Fully Autonomous and
Human-in-the-Loop (HITL) modes.

Design & Presentation Features:
-------------------------------
- Initial Load Splash Modal: Centered popup with cycling status text while the app hydates.
- Active Execution Modal: Centered popup with dynamic step-by-step phrases while running.
- High-Contrast Visible Input Box: 2px solid slate border permanently visible even unfocused.
- Refined Compact Typography: Smaller, tighter headings (19px h1, 16px h2, 14px h3).
- Advanced Action Button: Smooth gradient, elevation shadows, and micro-hover states.
- Zero Emojis: Clean typography, geometric status badges, and monochrome indicators.
- Preserved Functional Logic: Streaming, checkpointing, and interrupt/resume intact.
"""

import os
import re
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from config import get_secret, validate_gemini_key, validate_tavily_key
from agent.graph import run_newsletter_agent, resume_newsletter_agent, stream_newsletter_agent
from tools.html_generator import markdown_to_email_html

# Load environment configuration
load_dotenv()

# Configure Streamlit page metadata and layout
st.set_page_config(
    page_title="Newsletter Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(ttl=60, show_spinner=False)
def check_environment_status():
    """Validates Gemini and Tavily keys live using minimal test calls, cached for 60 seconds."""
    gem_ok, gem_msg = validate_gemini_key()
    tav_ok, tav_msg = validate_tavily_key()
    return gem_ok, gem_msg, tav_ok, tav_msg


# ==============================================================================
# Advanced CSS Design System (Refined Typography & High-Contrast Input)
# ==============================================================================

ADVANCED_CSS = """
<style>
/* --------------------------------------------------------------------------
   Root Variables & Color System
   -------------------------------------------------------------------------- */
:root {
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif;
  --bg-app: #F8FAFC;
  --bg-card: #FFFFFF;
  --bg-subtle: #F1F5F9;
  --border-subtle: #E2E8F0;
  --border-strong: #64748B;
  --border-focus: #2563EB;
  --text-primary: #0F172A;
  --text-secondary: #334155;
  --text-muted: #64748B;
  --primary-gradient: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #4F46E5 100%);
  --primary-hover: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 50%, #4338CA 100%);
  --success-bg: #ECFDF5;
  --success-text: #065F46;
  --success-border: #A7F3D0;
  --warning-bg: #FFFBEB;
  --warning-text: #92400E;
  --warning-border: #FDE68A;
  --danger-bg: #FEF2F2;
  --danger-text: #991B1B;
  --danger-border: #FECACA;
}

/* Force light mode regardless of OS/Browser theme preferences */
html, body, [data-testid="stAppViewContainer"], .main {
  background-color: var(--bg-app) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-sans) !important;
}

[data-testid="stSidebar"] {
  background-color: #FFFFFF !important;
  border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stHeader"] {
  background-color: transparent !important;
}

/* Refined, Compact Typography Hierarchy */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-sans) !important;
  color: var(--text-primary) !important;
  font-weight: 700 !important;
  letter-spacing: -0.015em !important;
}

h1 { font-size: 17px !important; margin-bottom: 2px !important; }
h2 { font-size: 15px !important; margin-top: 10px !important; margin-bottom: 4px !important; }
h3 { font-size: 13px !important; margin-top: 8px !important; margin-bottom: 3px !important; }
h4 { font-size: 12px !important; }

/* --------------------------------------------------------------------------
   Hero Card Container (Compact & Refined)
   -------------------------------------------------------------------------- */
.hero-card {
  background: #FFFFFF;
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 14px 20px;
  margin-bottom: 14px;
  box-shadow: 0 1px 6px rgba(15, 23, 42, 0.03);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.hero-title {
  font-size: 17px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.015em;
  margin: 0;
}

.hero-subtitle {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 2px;
  margin-bottom: 0;
  line-height: 1.35;
}

/* --------------------------------------------------------------------------
   Permanently Visible High-Contrast Input Box
   -------------------------------------------------------------------------- */
div[data-testid="stTextArea"] {
  margin-top: 4px !important;
  margin-bottom: 10px !important;
}

div[data-testid="stTextArea"] label[data-testid="stWidgetLabel"] p {
  font-size: 11.5px !important;
  font-weight: 700 !important;
  color: #1E293B !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
  margin-bottom: 4px !important;
}

div[data-testid="stTextArea"] > div,
div[data-testid="stTextArea"] div[data-baseweb="textarea"],
div[data-testid="stTextArea"] div[data-baseweb="base-input"] {
  border: 2px solid #475569 !important; /* Visible Slate-600 border even when unfocused */
  border-radius: 9px !important;
  background-color: #FFFFFF !important;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
  transition: all 0.2s ease !important;
}

div[data-testid="stTextArea"] > div:hover,
div[data-testid="stTextArea"] div[data-baseweb="textarea"]:hover {
  border-color: #1E293B !important;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12) !important;
}

div[data-testid="stTextArea"] > div:focus-within,
div[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within {
  border-color: #2563EB !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25) !important;
}

div[data-testid="stTextArea"] textarea {
  font-size: 13.5px !important;
  font-weight: 500 !important;
  line-height: 1.5 !important;
  color: #0F172A !important;
  background-color: #FFFFFF !important;
  padding: 10px 12px !important;
}

/* Quick Preset Chip Buttons */
div[data-testid="column"] div[data-testid="stButton"] > button {
  background: #F8FAFC !important;
  border: 1px solid #CBD5E1 !important;
  border-radius: 20px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  color: #334155 !important;
  padding: 5px 12px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
  transition: all 0.15s ease !important;
}

div[data-testid="column"] div[data-testid="stButton"] > button:hover {
  background: #EFF6FF !important;
  border-color: #3B82F6 !important;
  color: #1E40AF !important;
  box-shadow: 0 2px 6px rgba(59, 130, 246, 0.15) !important;
  transform: translateY(-1px) !important;
}

/* --------------------------------------------------------------------------
   Advanced Run Agent Button
   -------------------------------------------------------------------------- */
div.stButton > button[kind="primary"],
div[data-testid="column"]:last-child div.stButton > button[kind="primary"] {
  background: var(--primary-gradient) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 10px 24px !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
  cursor: pointer !important;
}

div.stButton > button[kind="primary"]:hover {
  background: var(--primary-hover) !important;
  box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
  transform: translateY(-1px) !important;
}

div.stButton > button[kind="primary"]:active {
  transform: scale(0.99) !important;
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
}

/* Secondary Buttons */
div.stButton > button:not([kind="primary"]) {
  background-color: #FFFFFF !important;
  color: var(--text-secondary) !important;
  border: 1px solid #CBD5E1 !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 7px 14px !important;
  transition: all 0.15s ease-in-out !important;
}

div.stButton > button:not([kind="primary"]):hover {
  background-color: var(--bg-subtle) !important;
  border-color: #94A3B8 !important;
  color: var(--text-primary) !important;
}

/* Download Buttons */
[data-testid="stDownloadButton"] > button {
  background-color: #FFFFFF !important;
  color: var(--text-primary) !important;
  border: 1px solid #CBD5E1 !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 8px 16px !important;
  transition: all 0.15s ease-in-out !important;
}

[data-testid="stDownloadButton"] > button:hover {
  background-color: var(--bg-subtle) !important;
  border-color: #2563EB !important;
  color: #2563EB !important;
}

/* --------------------------------------------------------------------------
   Small Inline Pipeline Status Indicators
   -------------------------------------------------------------------------- */
.compact-loader-card {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #EFF6FF;
  border: 1px solid #BFDBFE;
  border-radius: 8px;
  padding: 10px 14px;
  margin: 10px 0;
}

.compact-spinner {
  width: 18px;
  height: 18px;
  border: 2px solid #DBEAFE;
  border-top-color: #2563EB;
  border-radius: 50%;
  animation: compactSpin 0.75s linear infinite;
  flex-shrink: 0;
}

@keyframes compactSpin {
  to { transform: rotate(360deg); }
}

.compact-loader-title {
  font-size: 13px;
  font-weight: 700;
  color: #1E40AF;
  line-height: 1.2;
}

.compact-loader-desc {
  font-size: 11.5px;
  color: #475569;
  line-height: 1.2;
  margin-top: 1px;
}

/* --------------------------------------------------------------------------
   Status Badges / Pills
   -------------------------------------------------------------------------- */
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 9px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 600;
  line-height: 1.4;
  letter-spacing: 0.02em;
}

.status-pill::before {
  content: "";
  display: inline-block;
  width: 5px;
  height: 5px;
  border-radius: 50%;
}

.status-pill.success {
  background-color: var(--success-bg);
  color: var(--success-text);
  border: 1px solid var(--success-border);
}
.status-pill.success::before { background-color: #10B981; }

.status-pill.warning {
  background-color: var(--warning-bg);
  color: var(--warning-text);
  border: 1px solid var(--warning-border);
}
.status-pill.warning::before { background-color: #F59E0B; }

.status-pill.danger {
  background-color: var(--danger-bg);
  color: var(--danger-text);
  border: 1px solid var(--danger-border);
}
.status-pill.danger::before { background-color: #EF4444; }

.status-pill.neutral {
  background-color: var(--bg-subtle);
  color: var(--text-secondary);
  border: 1px solid #CBD5E1;
}
.status-pill.neutral::before { background-color: #64748B; }

/* --------------------------------------------------------------------------
   Horizontal Metric Statistics Row
   -------------------------------------------------------------------------- */
.stat-row {
  display: flex;
  gap: 12px;
  margin: 14px 0 20px 0;
  width: 100%;
}

.stat-box {
  flex: 1;
  background: #FFFFFF;
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 14px 18px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}

.stat-value {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.1;
  margin-bottom: 2px;
}

.stat-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
  color: var(--text-muted);
}

/* --------------------------------------------------------------------------
   Vertical Stepper Timeline
   -------------------------------------------------------------------------- */
.timeline {
  position: relative;
  padding-left: 26px;
  margin: 14px 0 6px 0;
}

.timeline::before {
  content: "";
  position: absolute;
  top: 6px;
  bottom: 6px;
  left: 7px;
  width: 2px;
  background: var(--border-subtle);
}

.timeline-item {
  position: relative;
  margin-bottom: 14px;
}

.timeline-item:last-child {
  margin-bottom: 0;
}

.timeline-dot {
  position: absolute;
  left: -26px;
  top: 4px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #FFFFFF;
  border: 2px solid #94A3B8;
  display: flex;
  align-items: center;
  justify-content: center;
}

.timeline-dot.success {
  border-color: #10B981;
  background: #10B981;
}

.timeline-dot.warning {
  border-color: #F59E0B;
  background: #F59E0B;
}

.timeline-dot.info {
  border-color: #2563EB;
  background: #2563EB;
}

.timeline-content {
  font-size: 13px;
  line-height: 1.45;
  color: var(--text-secondary);
}

.timeline-step-badge {
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-secondary);
  background: var(--bg-subtle);
  border: 1px solid var(--border-subtle);
  padding: 1px 6px;
  border-radius: 4px;
  margin-right: 6px;
}

/* --------------------------------------------------------------------------
   Mock Browser Chrome Frame
   -------------------------------------------------------------------------- */
.mock-browser-frame {
  border: 1px solid #CBD5E1;
  border-radius: 12px;
  overflow: hidden;
  background: #FFFFFF;
  margin-top: 10px;
  margin-bottom: 20px;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}

.mock-browser-chrome {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: var(--bg-subtle);
  border-bottom: 1px solid var(--border-subtle);
}

.mock-dots {
  display: flex;
  gap: 5px;
}

.mock-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
}

.mock-dot.red { background-color: #EF4444; }
.mock-dot.yellow { background-color: #F59E0B; }
.mock-dot.green { background-color: #10B981; }

.mock-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.mock-tag {
  font-size: 10px;
  font-weight: 600;
  color: #1E40AF;
  background: #EFF6FF;
  border: 1px solid #DBEAFE;
  padding: 1px 7px;
  border-radius: 9999px;
}
</style>
"""

st.markdown(ADVANCED_CSS, unsafe_allow_html=True)




# ==============================================================================
# Session State Initialization (Functional Logic Preserved)
# ==============================================================================

DEFAULT_GOAL = "Create a weekly newsletter on latest AI agent news and send it to our subscribers."

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = None

if "status" not in st.session_state:
    st.session_state["status"] = "idle"  # "idle" | "running" | "paused_for_review" | "complete" | "error"

if "agent_response" not in st.session_state:
    st.session_state["agent_response"] = None

if "request_changes_active" not in st.session_state:
    st.session_state["request_changes_active"] = False

if "error_message" not in st.session_state:
    st.session_state["error_message"] = None

if "goal_prompt" not in st.session_state:
    st.session_state["goal_prompt"] = DEFAULT_GOAL


# ==============================================================================
# Helper Renderers
# ==============================================================================

def render_timeline(stage_log: list[str]):
    """Renders stage_log entries as a connected vertical stepper timeline with timing badges."""
    if not stage_log:
        return

    items_html = []
    for idx, entry in enumerate(stage_log, 1):
        clean_entry = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf]', '', entry).strip()

        if "fail" in clean_entry.lower() or "error" in clean_entry.lower():
            dot_class = "warning"
        elif any(w in clean_entry.lower() for w in ["complete", "pass", "publish", "found", "written", "approved"]):
            dot_class = "success"
        else:
            dot_class = "info"

        # Extract timing bracket if present, e.g. [8.4s: ...] or [3.2s] or [Total pipeline time: 39.2s]
        timing_badge = ""
        timing_match = re.search(r'\[([^\]]+)\]$', clean_entry)
        display_text = clean_entry
        if timing_match:
            timing_val = timing_match.group(1).strip()
            display_text = clean_entry[:timing_match.start()].strip()
            timing_badge = (
                f'<span style="margin-left: 8px; font-size: 11px; font-weight: 600; '
                f'color: #2563EB; background: #EFF6FF; border: 1px solid #DBEAFE; '
                f'padding: 1px 7px; border-radius: 9999px;">{timing_val}</span>'
            )

        item_str = (
            f'<div class="timeline-item">'
            f'<div class="timeline-dot {dot_class}"></div>'
            f'<div class="timeline-content">'
            f'<span class="timeline-step-badge">Step {idx}</span>'
            f'<span>{display_text}</span>'
            f'{timing_badge}'
            f'</div>'
            f'</div>'
        )
        items_html.append(item_str)

    timeline_markup = (
        f'<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 22px; margin-top: 16px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">'
        f'<h3 style="margin: 0 0 10px 0; font-size: 14px; font-weight: 700; color: #1E293B;">Pipeline Execution History</h3>'
        f'<div class="timeline">{"".join(items_html)}</div>'
        f'</div>'
    )
    st.markdown(timeline_markup, unsafe_allow_html=True)


def render_critique_panel(critique: dict, revision_count: int):
    """Renders structured critique details with quality score and issues checklist."""
    if not critique:
        return

    passed = bool(critique.get("passed", False) or critique.get("pass", False))
    score = critique.get("score", "N/A")
    issues = critique.get("issues", [])
    suggestions = critique.get("suggestions", "")

    status_badge = (
        '<span class="status-pill success">Passed</span>'
        if passed
        else '<span class="status-pill warning">Revision Required</span>'
    )

    with st.expander(f"Editorial Critique Report — Iteration {revision_count}", expanded=True):
        critique_stat_html = (
            f'<div class="stat-row">'
            f'<div class="stat-box"><div class="stat-value">{score}/10</div><div class="stat-label">Quality Score</div></div>'
            f'<div class="stat-box"><div class="stat-value">{status_badge}</div><div class="stat-label" style="margin-top: 3px;">Verdict</div></div>'
            f'</div>'
        )
        st.markdown(critique_stat_html, unsafe_allow_html=True)

        if issues:
            st.markdown("**Identified Remediation Items:**")
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.markdown(
                '<div style="color: #065F46; font-size: 13px; padding: 4px 0;">'
                'All factual grounding and coverage standards satisfied.</div>',
                unsafe_allow_html=True,
            )

        if suggestions:
            st.markdown(f"**Editorial Suggestions:**\n> {suggestions}")


# ==============================================================================
# Header Card (Refined & Compact)
# ==============================================================================

st.markdown(
    """
    <div class="hero-card">
      <div>
        <h1 class="hero-title">Newsletter Agent</h1>
        <p class="hero-subtitle">
          Autonomous multi-step editorial agent orchestrating research, drafting, critique, and responsive HTML publishing.
        </p>
      </div>
      <div>
        <span class="status-pill neutral">LangGraph Orchestrated</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Sidebar: Controls & API Health Panel
# ==============================================================================

with st.sidebar:
    st.markdown("### Execution Configuration")

    mode_selection = st.radio(
        "Workflow Mode:",
        options=["Fully Autonomous", "Human-in-the-Loop"],
        key="workflow_mode_selection",
        help="Autonomous mode executes all stages automatically. Human-in-the-Loop pauses after critique for editorial review.",
    )
    mapped_mode = "human_in_loop" if mode_selection == "Human-in-the-Loop" else "autonomous"

    st.markdown("---")
    st.markdown("### Service Connectivity")

    gem_ok, gem_msg, tav_ok, tav_msg = check_environment_status()

    gem_badge = (
        '<span class="status-pill success">Connected</span>'
        if gem_ok
        else '<span class="status-pill danger">Disconnected</span>'
    )
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
          <span style="font-size: 13px; font-weight: 600; color: #334155;">Google Gemini API</span>
          {gem_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{gem_msg}")
    if not gem_ok:
        st.markdown("[Get Gemini API Key](https://aistudio.google.com/apikey)")

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    tav_badge = (
        '<span class="status-pill success">Connected</span>'
        if tav_ok
        else '<span class="status-pill danger">Disconnected</span>'
    )
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
          <span style="font-size: 13px; font-weight: 600; color: #334155;">Tavily Search API</span>
          {tav_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{tav_msg}")
    if not tav_ok:
        st.markdown("[Get Tavily API Key](https://tavily.com)")

    all_keys_valid = gem_ok and tav_ok

    if not all_keys_valid:
        st.markdown(
            """
            <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 10px; margin-top: 10px; font-size: 12px; color: #92400E;">
              API credentials missing. Configure keys in <code>.env</code> and test via <code>python scripts/check_setup.py</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Re-test Keys", use_container_width=True):
            check_environment_status.clear()
            st.rerun()
    with col_btn2:
        if st.button("Reset Session", use_container_width=True):
            st.session_state["thread_id"] = None
            st.session_state["status"] = "idle"
            st.session_state["agent_response"] = None
            st.session_state["request_changes_active"] = False
            st.session_state["error_message"] = None
            st.rerun()


# ==============================================================================
# Editorial Command Center (Permanently Visible Box with Presets)
# ==============================================================================

st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
      <span style="font-size: 12px; font-weight: 700; color: #1E293B; text-transform: uppercase; letter-spacing: 0.05em;">Editorial Command Center</span>
      <span style="font-size: 11px; color: #64748B;">Choose a quick preset or customize your instructions below</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Preset Topic Chips
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    if st.button("Frameworks & Orchestration", key="btn_preset_1", use_container_width=True):
        st.session_state["goal_prompt"] = (
            "Create a weekly newsletter on new autonomous AI agent frameworks, orchestration libraries, and open-source tooling."
        )
        st.rerun()
with col_p2:
    if st.button("Enterprise Deployments", key="btn_preset_2", use_container_width=True):
        st.session_state["goal_prompt"] = (
            "Cover enterprise autonomous agent deployments, production benchmarks, and infrastructure security protocols."
        )
        st.rerun()
with col_p3:
    if st.button("Funding & Market Moves", key="btn_preset_3", use_container_width=True):
        st.session_state["goal_prompt"] = (
            "Report on agentic AI startup venture rounds, valuation milestones, and commercial ecosystem partnerships."
        )
        st.rerun()

# Prominent, Permanently Visible Textarea
goal_input = st.text_area(
    label="Target Objective & Editorial Directives",
    value=st.session_state["goal_prompt"],
    height=95,
    help="Define the topic, target angle, or audience for the newsletter.",
)

# High-Impact Run Agent Button Bar
col_run_l, col_run_r = st.columns([3, 1])
with col_run_l:
    st.caption("Engine: Google Gemini 3.5 Flash-Lite + Tavily Search API • LangGraph Orchestrated")
with col_run_r:
    run_clicked = st.button(
        "Run Agent",
        type="primary",
        use_container_width=True,
        disabled=not all_keys_valid,
    )


# ==============================================================================
# Execution Flow & Active Loading Popup (Rotating Step Phrases)
# ==============================================================================

if run_clicked:
    if not all_keys_valid:
        st.error("Cannot run agent without valid credentials. Please configure .env.")
    else:
        st.session_state["status"] = "running"
        st.session_state["request_changes_active"] = False
        st.session_state["error_message"] = None

        # Small inline loader
        st.markdown(
            """
            <div class="compact-loader-card">
              <div class="compact-spinner"></div>
              <div>
                <div class="compact-loader-title">Autonomous Agent Active</div>
                <div class="compact-loader-desc">Executing editorial pipeline with Gemini 3.5 Flash-Lite &amp; Tavily</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.status("Agent Workflow in Progress...", expanded=True) as status_box:
            try:
                final_response = None
                for node_name, update, data in stream_newsletter_agent(goal=goal_input.strip(), mode=mapped_mode):
                    if node_name == "planner":
                        queries = update.get("search_queries", [])
                        status_box.write(f"Step 1: Editorial Strategy — Formulated {len(queries)} targeted search queries.")
                    elif node_name == "researcher":
                        r_count = len(update.get("raw_search_results", []))
                        s_count = len(update.get("research_summary", []))
                        status_box.write(f"Step 2: Grounded Retrieval — Gathered {r_count} items; synthesized {s_count} structured summaries.")
                    elif node_name == "writer":
                        subj = update.get("subject_line", "")
                        status_box.write(f"Step 3: Content Drafting — Generated newsletter draft ('{subj[:40]}...').")
                    elif node_name == "critic":
                        c = update.get("critique", {})
                        passed = c.get("passed", False)
                        score = c.get("score", "N/A")
                        verdict = "Passed" if passed else "Needs Revision"
                        status_box.write(f"Step 4: Editorial Critique — Score {score}/10 ({verdict}).")
                    elif node_name == "reviser":
                        rev = update.get("revision_count", 1)
                        status_box.write(f"Revision Cycle — Remediating issues (Attempt {rev})...")
                    elif node_name == "publisher":
                        status_box.write("Step 5: Publishing Deliverables — Inlined responsive CSS and saved deliverables.")
                    elif node_name in ["human_review", "interrupt"]:
                        status_box.write("Step 5: Human Review Checkpoint — Paused for editorial sign-off.")
                        if node_name == "interrupt":
                            final_response = data
                    elif node_name == "complete":
                        final_response = data

                if final_response:
                    st.session_state["thread_id"] = final_response.get("thread_id")
                    st.session_state["status"] = final_response.get("status")
                    st.session_state["agent_response"] = final_response
                    if final_response.get("status") == "paused_for_review":
                        status_box.update(label="Paused at Checkpoint — Editorial Approval Required", state="complete", expanded=False)
                    else:
                        status_box.update(label="Pipeline execution completed successfully", state="complete", expanded=False)
            except Exception as exc:
                st.session_state["status"] = "error"
                st.session_state["error_message"] = str(exc)
                status_box.update(label="Pipeline execution failed", state="error", expanded=True)

        st.rerun()


# ==============================================================================
# Error Display
# ==============================================================================

if st.session_state["status"] == "error":
    st.markdown(
        f"""
        <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px; padding: 14px 18px; margin: 18px 0; color: #991B1B; font-size: 13px;">
          <strong>Execution Failure:</strong> {st.session_state['error_message']}
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    preview_html = markdown_to_email_html(draft_md, subject)

    banner_html = (
        '<div style="background: #FFFBEB; border: 1px solid #FDE68A; border-left: 5px solid #D97706; border-radius: 10px; padding: 16px 20px; margin: 18px 0;">'
        '<div style="display: flex; justify-content: space-between; align-items: center;">'
        '<div>'
        '<span style="font-weight: 700; color: #92400E; font-size: 15px;">Editorial Sign-Off Required (Checkpoint Paused)</span>'
        '<p style="color: #B45309; font-size: 13px; margin: 3px 0 0 0;">'
        'The pipeline is paused and waiting for your review. Approve publication or request specific revisions using the controls below.'
        '</p>'
        '</div>'
        '<span class="status-pill warning" style="font-size: 12px; font-weight: 700; padding: 4px 12px;">Paused for Approval</span>'
        '</div>'
        '</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    state_data = response.get("state", {})
    stage_timings = state_data.get("stage_timings", {})
    elapsed_time = sum(v if isinstance(v, (int, float)) else sum(v) for k, v in stage_timings.items() if k not in ["research_search", "research_synthesis", "total"])
    elapsed_time_str = f"{elapsed_time:.1f}s" if elapsed_time > 0 else "N/A"
    critique_score = critique.get("score", "N/A")

    stat_row_html = (
        f'<div class="stat-row">'
        f'<div class="stat-box"><div class="stat-value">Paused</div><div class="stat-label">Pipeline Status</div></div>'
        f'<div class="stat-box"><div class="stat-value">{revision_count}</div><div class="stat-label">Revision Iterations</div></div>'
        f'<div class="stat-box"><div class="stat-value">{critique_score}/10</div><div class="stat-label">Draft Quality Score</div></div>'
        f'<div class="stat-box"><div class="stat-value">{elapsed_time_str}</div><div class="stat-label">Total Time</div></div>'
        f'</div>'
    )
    st.markdown(stat_row_html, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Editorial Decision Action Card (Placed Above Draft & Previews)
    # --------------------------------------------------------------------------
    decision_card_header = (
        '<div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 10px; padding: 14px 18px; margin: 16px 0 10px 0;">'
        '<div style="display: flex; justify-content: space-between; align-items: center;">'
        '<span style="font-size: 13px; font-weight: 700; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em;">Editorial Action Checkpoint</span>'
        '<span style="font-size: 11px; font-weight: 600; color: #2563EB; background: #EFF6FF; border: 1px solid #DBEAFE; padding: 2px 8px; border-radius: 9999px;">Action Required</span>'
        '</div>'
        '</div>'
    )
    st.markdown(decision_card_header, unsafe_allow_html=True)

    col_approve, col_reject = st.columns([1, 1])

    with col_approve:
        if st.button("Approve & Publish", type="primary", use_container_width=True):
            with st.spinner("Resuming graph execution with approval..."):
                try:
                    resume_res = resume_newsletter_agent(thread_id, {"decision": "approve"})
                    st.session_state["status"] = resume_res.get("status")
                    st.session_state["agent_response"] = resume_res
                    st.session_state["request_changes_active"] = False
                except Exception as exc:
                    st.session_state["status"] = "error"
                    st.session_state["error_message"] = str(exc)
            st.rerun()

    with col_reject:
        if st.button("Request Changes", use_container_width=True):
            st.session_state["request_changes_active"] = True
            st.rerun()

    if st.session_state.get("request_changes_active"):
        with st.form("feedback_form"):
            feedback_text = st.text_area(
                "Editorial Directives / Specific Modifications:",
                placeholder="E.g., Strengthen the technical takeaways in Story 2, and tone down promotional wording in the intro...",
                height=110,
            )
            submit_feedback = st.form_submit_button("Submit Feedback & Revise", type="primary")

            if submit_feedback:
                if not feedback_text.strip():
                    st.warning("Please supply feedback instructions before submitting.")
                else:
                    with st.spinner("Resuming agent with revision directives..."):
                        try:
                            resume_res = resume_newsletter_agent(
                                thread_id,
                                {"decision": "request_changes", "feedback": feedback_text.strip()},
                            )
                            st.session_state["status"] = resume_res.get("status")
                            st.session_state["agent_response"] = resume_res
                            st.session_state["request_changes_active"] = False
                        except Exception as exc:
                            st.session_state["status"] = "error"
                            st.session_state["error_message"] = str(exc)
                    st.rerun()

    # Render Critique Report
    render_critique_panel(critique, revision_count)

    stage_log = payload.get("stage_log") or state_data.get("stage_log", [])
    if stage_log:
        render_timeline(stage_log)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown(f"#### Draft Subject Line: **{subject}**")

    # Framed Browser Container for Live Preview
    preview_tabs = st.tabs(["Rendered Email Preview", "Raw Markdown Draft"])

    with preview_tabs[0]:
        mock_chrome_html = (
            '<div class="mock-browser-frame">'
            '<div class="mock-browser-chrome">'
            '<div class="mock-dots">'
            '<span class="mock-dot red"></span>'
            '<span class="mock-dot yellow"></span>'
            '<span class="mock-dot green"></span>'
            '</div>'
            '<div class="mock-title">Draft Preview — Pending Approval</div>'
            '<div class="mock-tag" style="color: #92400E; background: #FEF3C7; border: 1px solid #FDE68A;">Unapproved Draft</div>'
            '</div>'
            '</div>'
        )
        st.markdown(mock_chrome_html, unsafe_allow_html=True)
        components.html(preview_html, height=560, scrolling=True)

    with preview_tabs[1]:
        st.markdown(draft_md)


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

    st.markdown(
        """
        <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 10px; padding: 16px 22px; margin-bottom: 20px;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <span style="font-weight: 700; color: #065F46; font-size: 15px;">Newsletter Published Successfully</span>
              <p style="color: #047857; font-size: 13px; margin: 3px 0 0 0;">
                All verification criteria met. Responsive email HTML and markdown deliverables generated.
              </p>
            </div>
            <span class="status-pill success">Published</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage_timings = result.get("stage_timings", {})
    total_time_val = stage_timings.get("total")
    if not total_time_val:
        total_time_val = sum(v if isinstance(v, (int, float)) else sum(v) for k, v in stage_timings.items() if k not in ["research_search", "research_synthesis", "total"])
    total_time_str = f"{total_time_val:.1f}s" if total_time_val else "N/A"

    # Clean Horizontal Metrics Row
    stat_row_complete_html = (
        f'<div class="stat-row">'
        f'<div class="stat-box"><div class="stat-value">Published</div><div class="stat-label">Pipeline Status</div></div>'
        f'<div class="stat-box"><div class="stat-value">{revision_count}</div><div class="stat-label">Revision Iterations</div></div>'
        f'<div class="stat-box"><div class="stat-value">{critique_score}/10</div><div class="stat-label">Final Quality Score</div></div>'
        f'<div class="stat-box"><div class="stat-value">{total_time_str}</div><div class="stat-label">Total Time</div></div>'
        f'</div>'
    )
    st.markdown(stat_row_complete_html, unsafe_allow_html=True)

    st.markdown(f"### Subject: **{subject_line}**")

    # Deliverables Tabs
    deliverable_tabs = st.tabs(["Rendered Email Output", "Markdown Content", "Saved Artifacts"])

    with deliverable_tabs[0]:
        st.markdown(
            """
            <div class="mock-browser-frame">
              <div class="mock-browser-chrome">
                <div class="mock-dots">
                  <span class="mock-dot red"></span>
                  <span class="mock-dot yellow"></span>
                  <span class="mock-dot green"></span>
                </div>
                <div class="mock-title">Preview — Responsive HTML Email</div>
                <div class="mock-tag">Final Deliverable</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        components.html(final_html, height=600, scrolling=True)

    with deliverable_tabs[1]:
        st.markdown(final_md)

    with deliverable_tabs[2]:
        st.markdown(f"**Local File Path:** `{output_path}`")
        if os.path.exists(output_path):
            st.markdown(
                f'<div style="color: #475569; font-size: 13px;">Verified on disk ({os.path.getsize(output_path):,} bytes).</div>',
                unsafe_allow_html=True,
            )

    # Export Deliverables Section
    st.markdown("---")
    st.markdown("### Export Deliverables")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        st.download_button(
            label="Download HTML Email (.html)",
            data=final_html,
            file_name=os.path.basename(output_path) if output_path else "newsletter.html",
            mime="text/html",
            use_container_width=True,
        )

    with dl_col2:
        st.download_button(
            label="Download Markdown Draft (.md)",
            data=final_md,
            file_name="newsletter.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Historical Stepper & Critique
    st.markdown("---")
    render_critique_panel(critique, revision_count)
    render_timeline(stage_log)
