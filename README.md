# Newsletter Agent

An autonomous multi-step editorial agent that researches, drafts, critiques, revises, and publishes an AI agent newsletter with an Autonomous / Human-in-the-Loop toggle.

**Live Demo:** [assignment-ai-agent.streamlit.app](https://assignment-ai-agent.streamlit.app/)

---

## Architecture

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

---

## Core Features

- **Multi-step reasoning**: 7 specialized LangGraph nodes execute planning, retrieval, drafting, critique, revision, human review, and publication.
- **Tool use**: Decoupled tools for Tavily web search, Gemini structured research synthesis, and inline HTML email compilation.
- **Self-critique revision loop**: Critic node evaluates drafts against factuality and coverage rubrics, routing back to reviser/writer on failure (max 2 cycles).
- **Autonomous retry**: Researcher auto-broadens date windows (30d → 90d → 180d) if initial queries return zero results.
- **Human-in-the-Loop toggle**: Native LangGraph `interrupt()` / `resume()` checkpoint enables approving drafts or requesting targeted revisions before publication.
- **Stage timing instrumentation**: Measures search API vs. LLM synthesis latency per stage, displayed in logs and UI timeline.

---

## Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | Cyclic state machine, routing, and interrupt checkpoints |
| **LLM Engine** | [Google Gemini](https://aistudio.google.com/) (`gemini-3.5-flash-lite`) | Planning, structured research synthesis, drafting, critique |
| **Search API** | [Tavily AI](https://tavily.com/) | Real-time web retrieval with date windowing |
| **Frontend** | [Streamlit](https://streamlit.io/) | Streaming execution stepper, review controls, HTML preview |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) | Structured output schemas for node inputs/outputs |

---

## Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/singhparmeet12/Assignment-AI-agent.git
   cd Assignment-AI-agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment keys**
   ```bash
   cp .env.example .env
   ```
   Add your `GOOGLE_API_KEY` and `TAVILY_API_KEY` to `.env`.

5. **Verify setup**
   ```bash
   python scripts/check_setup.py
   ```

---

## Run

- **Streamlit Web UI:**
  ```bash
  streamlit run app.py
  ```

- **CLI:**
  ```bash
  # Autonomous mode:
  python main.py --mode autonomous --goal "Create a weekly newsletter on latest AI agent news"

  # Human-in-the-Loop mode:
  python main.py --mode hitl --goal "Create a weekly newsletter on latest AI agent news"
  ```

---

## Project Structure

```
├── agent/
│   ├── graph.py             # LangGraph state machine & entry points
│   ├── nodes.py             # Pipeline node implementations
│   └── state.py             # NewsletterState schema definition
├── config.py                # Environment configuration (.env & st.secrets)
├── outputs/                 # Generated .html and .md newsletter deliverables
├── scripts/
│   ├── check_setup.py       # API authentication test
│   └── debug_hitl.py        # Headless HITL verification script
├── tools/
│   ├── html_generator.py    # Markdown to responsive HTML email converter
│   ├── search.py            # Tavily news search wrapper
│   └── summarizer.py        # Gemini structured synthesis
├── app.py                   # Streamlit web application
├── main.py                  # CLI entry point
└── requirements.txt
```
