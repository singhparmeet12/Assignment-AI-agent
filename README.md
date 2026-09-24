# 📰 Newsletter Agent

> An autonomous multi-step AI agent that researches, drafts, critiques, revises, and publishes an AI-agent-news newsletter, with an Autonomous / Human-in-the-Loop toggle.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LLM](https://img.shields.io/badge/Model-Google%20Gemini-4285F4.svg)](https://aistudio.google.com/)
[![Search](https://img.shields.io/badge/Search-Tavily%20AI-green.svg)](https://tavily.com/)
[![UI](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 1. Project Title & Description

**Newsletter Agent** is a production-grade, state-machine-driven editorial pipeline designed to autonomously monitor, curate, write, fact-check, and publish high-quality weekly newsletters focused on artificial intelligence agents and multi-agent systems. Built with **LangGraph**, **Google Gemini**, and **Tavily Search**, it balances deep autonomy with granular human oversight through a native **Human-in-the-Loop (HITL)** editorial checkpoint.

---

## 2. Architecture Overview

### State Machine Topology

The agent is organized as a cyclic state machine with explicit routing rules and dynamic feedback loops:

```
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
```

### Why LangGraph?

Traditional linear agent chains (or basic LLM tool-calling wrappers) execute sequentially without deterministic state boundaries. If an intermediate tool fails, or if a generated draft contains factual errors, a linear script either crashes or produces hollow content. 

LangGraph was specifically chosen because:
1. **Explicit State Machine**: The state schema (`NewsletterState`) is strictly typed, deterministic, and inspectable at each stage transition.
2. **First-Class Cyclic Loops**: It provides native conditional edges for self-correction loops (research retries and writer-critic revision cycles) without complex recursion hacks.
3. **Stateful Checkpointing & True HITL**: Through `MemorySaver` and the native `interrupt()` / `resume()` primitives, the pipeline can freeze execution mid-flight, yield control to a human reviewer in the UI or CLI, and cleanly resume execution without re-running earlier nodes or losing conversational context.

---

## 3. Core Agentic Capabilities

This project implements all core agentic design patterns outlined in modern AI engineering:

### A. Multi-Step Reasoning (7-Node Pipeline)
The pipeline breaks down open-ended editorial directives into 7 discrete, specialized graph nodes:

1. **`planner_node`**: Decomposes the user's objective into overarching editorial narrative pillars and formulates 3–5 targeted, high-precision search queries.
2. **`researcher_node`**: Executes web retrieval via Tavily and synthesizes findings into 5–7 structured, fact-checked summaries via Gemini.
3. **`writer_node`**: Transforms curated research summaries and the editorial plan into an authoritative, publication-ready markdown newsletter with verified source citations.
4. **`critic_node`**: Evaluates the draft against a strict rubric covering factual grounding, story coverage (5–7 stories), narrative flow, and structural integrity.
5. **`reviser_node`**: Performs revision bookkeeping, tracks feedback history, increments revision counters, and routes back to `writer_node`.
6. **`human_review_node`**: Serves as the interactive checkpoint in Human-in-the-Loop mode, suspending graph execution via `interrupt()` for human editorial sign-off.
7. **`publisher_node`**: Compiles the approved markdown into an inline-styled, responsive HTML email template, exports artifacts to disk (`outputs/`), and simulates distribution.

### B. Tool Use (3 Specialized Tools)
The agent leverages three decoupled tools that operate independently of graph state:
- **`tools/search.py` (`search_news`)**: Wraps Tavily Search with domain-specific date windowing, deduplication by canonical URL, and per-query raw result accounting.
- **`tools/summarizer.py` (`synthesize_research`)**: Uses Gemini structured outputs to filter fluff, consolidate duplicate coverage, enforce a 5–7 story target, and log an explicit accounting breakdown of dropped items.
- **`tools/html_generator.py` (`markdown_to_email_html`)**: Translates markdown into publication-grade HTML featuring inlined CSS, high-contrast dark badges, styled blockquotes, verified source buttons, and email-client-compatible footers.

### C. Self-Reflection & Critique Loop
- **Critic & Reviser Nodes**: The draft is rigorously reviewed by `critic_node`. If issues are detected (e.g., weak transitions, ungrounded claims, missing links), it generates structured actionable feedback and routes to `reviser_node` $\to$ `writer_node`.
- **Targeted Revision Directives**: In revision cycles, `writer_node` does not regenerate from scratch; it receives the previous draft, the specific critique issues, and concrete instructions, preserving verified content while resolving flaws.
- **`max_revisions` Safety Cap**: To prevent infinite execution loops and runaway API costs, a hard threshold (`max_revisions=2`) guarantees termination, forwarding to human review or publishing even if minor stylistic critique warnings remain.

### D. Self-Correcting Research Retry & Autonomous Recovery
- **Autonomous Parameter Expansion**: If an initial research query returns 0 results, `researcher_node` does not crash the pipeline. It automatically executes a self-correction loop:
  - **Attempt 1**: Standard 30-day news search.
  - **Attempt 2**: Widens search window to 90 days.
  - **Attempt 3**: Widens window to 180 days and simplifies queries to core technical keywords.
- **Honest-Failure Fallback**: If research completely fails after 3 attempts, `writer_node` refuses to hallucinate fake news stories. Instead, it generates an honest "Research Hiatus & Ecosystem Check" bulletin stating that no verified sources met standards, ensuring zero hallucinations.

### E. Human-in-the-Loop (HITL) Toggle
- **`mode="autonomous"`**: The agent plans, researches, drafts, self-critiques, revises, and publishes end-to-end without stopping.
- **`mode="human_in_loop"`**: Upon passing critique, conditional routing sends the draft to `human_review_node`. The graph suspends via LangGraph's native `interrupt()`. The user reviews the draft, subject line, and critique score in the UI/CLI, choosing to either:
  - **Approve**: Resumes graph to `publisher_node` to export the final newsletter.
  - **Request Changes**: Supplies custom feedback notes; the graph resumes to `reviser_node` and `writer_node` to apply the requested edits.

---

## 4. Tech Stack

| Component | Technology | Version / Model | Role in Architecture |
| :--- | :--- | :--- | :--- |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | `^0.2.70` | Stateful cyclic graph execution, routing, checkpointing & interrupts |
| **LLM Engine** | [Google Gemini](https://aistudio.google.com/) | `gemini-3.5-flash-lite` | Planning, research synthesis, draft generation, and editorial critique |
| **Web Search** | [Tavily AI](https://tavily.com/) | `tavily-python ^0.5.1` | Real-time, grounded search engine designed for AI agents |
| **Web Frontend** | [Streamlit](https://streamlit.io/) | `^1.42.0` | Real-time visual progress monitoring, review modal, and HTML preview |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) | `^2.10.6` | Schema enforcement for structured LLM inputs/outputs |
| **HTML Compilation** | Python Markdown | `^3.7` | AST markdown transformation and inline email styling |

---

## 5. Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/singhparmeet12/Assignment-AI-agent.git
cd Assignment-AI-agent
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.\.venv\Scripts\activate

# Activate on Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Copy the example environment configuration:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and fill in your keys:
- **`GOOGLE_API_KEY`**: Acquire free from [Google AI Studio](https://aistudio.google.com/apikey).
- **`TAVILY_API_KEY`**: Acquire free from [Tavily AI](https://tavily.com/).
- **`GEMINI_MODEL`**: Set to `gemini-3.5-flash-lite` (recommended for optimal speed and quota resilience).

### 5. Validate Configuration (Pre-Flight Check)
Run the automated diagnostic utility to verify connectivity before launching the agent:
```bash
python scripts/check_setup.py
```
*The script makes real minimal calls to Gemini and Tavily. It will exit with code `0` when all credentials are valid.*

---

## 6. Running the Project

### Option A: Interactive Web UI (Streamlit)
```bash
streamlit run app.py
```
- Open your browser at `http://localhost:8501`.
- Monitor live execution status via expanding phase containers (`st.status`).
- Review drafts and execute Human-in-the-Loop approvals directly in the browser.
- Preview rendered, email-client-ready HTML with responsive styling.

### Option B: Command-Line Interface (CLI)

- **Autonomous Mode (Hands-free execution):**
  ```bash
  python main.py --mode autonomous --goal "Create a weekly newsletter on latest AI agent news"
  ```

- **Human-in-the-Loop Mode (Interactive terminal review):**
  ```bash
  python main.py --mode hitl --goal "Create a weekly newsletter on latest AI agent news"
  ```

---

## 7. Demo / Sample Runs

The system was evaluated in real-world test scenarios, validating its resilience and self-healing properties:

### Observed Run A: Self-Critique Revision Loop & Rubric Refinement
- **Scenario**: During an initial test run with 5 available summaries, the agent drafted a 5-story newsletter.
- **Observation**: The draft was critiqued and failed because the legacy rubric required "an additional story" despite only 5 summaries existing in research. The graph cleanly routed to `reviser_node` $\to$ `writer_node`, illustrating the cyclic loop in action.
- **Outcome**: This real test case helped surface and fix an edge case in `critic_node`—aligning the rubric to treat 5 stories as a valid passing floor when all available research is utilized.

### Observed Run B: Clean First-Pass Production Run
- **Scenario**: Tested with 18 raw search results from Tavily.
- **Execution Flow**:
  1. `planner_node` generated 4 targeted queries.
  2. `researcher_node` retrieved 18 raw articles and synthesized **7 distinct, high-impact stories**.
  3. `writer_node` crafted a publication-grade markdown newsletter covering all 7 developments.
  4. `critic_node` evaluated the draft: **PASS on iteration 0 (Score: 10/10, 0 issues)**.
  5. `publisher_node` compiled and saved final `.md` and `.html` deliverables.

### Generated Artifacts
Real artifacts from actual runs are stored in the [`outputs/`](file:///c:/Parmeet/AI%20Agent%20Assignment/outputs) directory, including:
- `outputs/newsletter_20260924_235318.md`: Complete markdown draft covering 7 major developments (AWS Strands Harness, Temporal Series E, Cohesity Agent Resilience, etc.).
- `outputs/newsletter_20260924_235318.html`: Full responsive HTML email format with inlined CSS, source buttons, and headers.

---

## 8. Deployment

- **Hosted Platform**: **Streamlit Community Cloud**
- **Architecture Note**: Serverless environments (e.g., standard Vercel or AWS Lambda functions) are poorly suited for stateful LangGraph workflows because the in-memory checkpointing (`MemorySaver`) and `interrupt()` / `resume()` mechanics require a persistent Python execution process to wait for human editorial decisions. Streamlit Community Cloud maintains long-running sessions ideal for this architecture.
- **Live Demo URL**: `https://assignment-ai-agent.streamlit.app/` *(or local at `http://localhost:8501`)*

---

## 9. Project Structure

```
AI Agent Assignment/
├── agent/
│   ├── graph.py             # LangGraph orchestration, topology definition, checkpointing & runner functions
│   ├── nodes.py             # 7 specialized pipeline nodes (planner, researcher, writer, critic, etc.)
│   └── state.py             # Typed NewsletterState definition and initial state constructor
├── config.py                # Environment helper with support for .env and st.secrets
├── outputs/                 # Directory holding real generated .md and .html newsletter deliverables
│   ├── newsletter_*.html    # Compiled email HTML files
│   └── newsletter_*.md      # Markdown draft files
├── requirements.txt         # Core dependencies (langgraph, langchain-google-genai, tavily, streamlit)
├── scripts/
│   └── check_setup.py       # Pre-flight diagnostic script for live API key authentication
├── tools/
│   ├── html_generator.py    # AST markdown-to-email HTML converter with inline styles
│   ├── search.py            # Grounded Tavily search wrapper with date windowing & query logging
│   └── summarizer.py        # Gemini-powered structured research synthesis & deduplication
├── app.py                   # Streamlit web interface with real-time streaming & review modals
├── main.py                  # CLI runner supporting autonomous and HITL modes
├── .env.example             # Clean template for API credentials
└── README.md                # Project documentation and architectural guide
```

---

## 10. Design Decisions & Engineering Challenges

During the engineering of this agent, real-world execution revealed two subtle failure modes that were resolved:

1. **Research Synthesis Under-Generation (Bug 1)**:
   - *Problem*: Given 17 high-quality raw search results, `synthesize_research()` returned only 5 structured summaries despite `top_n=7`. The LLM defaulted to the minimum bound when presented with permissive prompt phrasing.
   - *Resolution*: Enforced strict target-quantity directives instructing the model to synthesize between 5 and 7 stories (prioritizing 7 whenever distinct topics exist), added structured `dropped_reasoning` tracking, and implemented a programmatic $\ge 5$ floor.

2. **Unsatisfiable Critique Loop (Bug 2)**:
   - *Problem*: In runs where 5 summaries were curated, the writer produced a 5-story draft. The critic rejected the draft for *"covering only 5 stories"* and demanded an additional story. Because the writer only had 5 source summaries, it was impossible to add a 6th story without hallucinating ungrounded facts, causing repeated failed revision loops.
   - *Resolution*: Updated `CRITIC_SYSTEM_PROMPT` to recognize 5, 6, or 7 stories as a complete pass, explicitly passed `available_count = len(research_summary)` into the critique context, and added a programmatic safeguard in `critic_node` that strips false-negative coverage rejections.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
