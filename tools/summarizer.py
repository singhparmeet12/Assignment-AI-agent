"""
tools/summarizer.py: Research Synthesis and Article Summarization Tool.

This module processes raw search results through an LLM to rank, filter,
deduplicate, and extract key insights into a clean, structured research digest.
It operates independently of LangGraph state structures.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

# Ensure workspace root is in sys.path when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_secret

# Configure module logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SynthesizedArticle(BaseModel):
    """Pydantic model representing a single synthesized research article."""
    title: str = Field(
        ...,
        description="Clear, engaging headline summarizing the core development or event."
    )
    summary: str = Field(
        ...,
        description="2-4 sentence analytical summary highlighting the key facts, impact, and takeaways."
    )
    source_url: str = Field(
        ...,
        description="Exact URL of the primary source article."
    )
    published_date: Optional[str] = Field(
        default=None,
        description="Publication date (e.g., '2025-05-10', 'March 2025', or 'Recent') if available."
    )


class ResearchSynthesis(BaseModel):
    """Container model for structured LLM research output."""
    articles: list[SynthesizedArticle] = Field(
        default_factory=list,
        description=(
            "List of distinct, high-impact synthesized articles. "
            "When sufficient raw results exist (e.g. 10+ items), you MUST provide between 5 and 7 stories, "
            "prioritizing filling toward 7 whenever distinct material exists."
        )
    )
    dropped_reasoning: Optional[str] = Field(
        default="",
        description=(
            "Concise breakdown of why any raw items were excluded "
            "(e.g., '10 items excluded: 4 duplicate coverage of framework X, 3 off-topic, 3 low-substance fluff')."
        )
    )


def synthesize_research(
    raw_results: list[dict],
    llm: Any,
    top_n: int = 7,
    focus_topic: str = "AI agent and generative AI developments"
) -> list[dict]:
    """
    Synthesizes and ranks raw search findings using an LLM. Discards low-value
    content and duplicate stories, returning between 5 and top_n structured articles.

    Args:
        raw_results: List of search result dictionaries containing 'title', 'url', 'snippet', etc.
        llm: A LangChain ChatModel instance (e.g., ChatGoogleGenerativeAI).
        top_n: Target number of top items to return (default: 7).
        focus_topic: The topical focus to guide relevance ranking.

    Returns:
        list[dict]: A list of dicts with keys:
            - "title": str
            - "summary": str
            - "source_url": str
            - "published_date": Optional[str]
    """
    if not raw_results:
        logger.warning("synthesize_research received empty raw_results list.")
        return []

    num_raw = len(raw_results)

    # Clean and prepare input payload for the LLM prompt
    formatted_items = []
    for idx, item in enumerate(raw_results, 1):
        formatted_items.append(
            f"[{idx}] Title: {item.get('title', 'Untitled')}\n"
            f"     URL: {item.get('url', '')}\n"
            f"     Date: {item.get('published_date', 'Unknown')}\n"
            f"     Snippet: {item.get('snippet', '')}"
        )
    articles_corpus = "\n\n".join(formatted_items)

    system_prompt = (
        "You are an expert Senior Editorial Research Analyst for an elite technology newsletter.\n"
        f"Your task is to analyze {num_raw} raw search results about '{focus_topic}', evaluate their quality, "
        f"and curate between 5 and {top_n} distinct, high-impact stories. Aim for exactly {top_n} stories whenever distinct developments exist.\n\n"
        "STRICT STORY QUANTITY & SELECTION RULES:\n"
        f"1. Target Quantity: There are {num_raw} raw results available. When {num_raw} results exist (especially 10+), you MUST synthesize between 5 and {top_n} distinct stories.\n"
        f"   - Prioritize filling toward {top_n} stories (i.e. 7 stories) when there is enough genuinely distinct source material.\n"
        f"   - Do NOT prematurely stop at 5 stories if 6 or 7 distinct topics, frameworks, benchmarks, or developments are present in the raw results.\n"
        f"   - 5 or 6 stories is acceptable ONLY if deduplication or quality filtering genuinely leaves fewer than {top_n} distinct items.\n"
        f"   - NEVER return fewer than 5 stories when at least 5 raw results are provided.\n"
        "2. Genuine Distinction: Each story MUST focus on a distinct development, framework, benchmark, funding event, or enterprise deployment. "
        "Do NOT create multiple stories covering the exact same announcement.\n"
        "3. Deduplication: If multiple search items report on the same event, select the best canonical source URL, merge facts into one strong summary, and do not repeat that story.\n"
        "4. Quality Filter: Discard purely off-topic, spammy, or marketing-only fluff. However, do NOT discard substantive technical news simply to reduce the count.\n"
        "5. Output Content per Story:\n"
        "   - title: Crisp, informative headline.\n"
        "   - summary: 2-4 sentences explaining what happened, why it matters, and practitioner takeaways.\n"
        "   - source_url: Exact URL from raw search results.\n"
        "   - published_date: Publication date string if available.\n"
        "6. In 'dropped_reasoning', provide an explicit breakdown of how many raw items were dropped and why (duplicate, off-topic, or low quality)."
    )

    user_prompt = (
        f"Raw Search Results ({num_raw} total):\n\n{articles_corpus}\n\n"
        f"CRITICAL INSTRUCTION: With {num_raw} raw results provided, you MUST curate and synthesize "
        f"between 5 and {top_n} distinct stories. Prioritize filling toward {top_n} stories (target: 7) if distinct material exists. "
        "Provide your dropped_reasoning explaining how many raw results were dropped and why."
    )

    output_list: list[dict] = []
    dropped_reasoning = ""

    if llm is None:
        logger.warning("No LLM client provided. Using raw search fallback formatting.")
    else:
        # Attempt structured output generation using LangChain
        try:
            if hasattr(llm, "with_structured_output"):
                structured_chain = llm.with_structured_output(ResearchSynthesis)
                result = structured_chain.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])

                if isinstance(result, ResearchSynthesis):
                    dropped_reasoning = result.dropped_reasoning or ""
                    output_list = [
                        {
                            "title": art.title.strip(),
                            "summary": art.summary.strip(),
                            "source_url": art.source_url.strip(),
                            "published_date": art.published_date,
                        }
                        for art in result.articles[:top_n]
                        if art.title.strip() and art.source_url.strip()
                    ]
                elif isinstance(result, dict) and "articles" in result:
                    dropped_reasoning = result.get("dropped_reasoning", "")
                    output_list = [
                        {
                            "title": art.get("title", "").strip(),
                            "summary": art.get("summary", "").strip(),
                            "source_url": art.get("source_url", "").strip(),
                            "published_date": art.get("published_date"),
                        }
                        for art in result["articles"][:top_n]
                        if art.get("title", "").strip() and art.get("source_url", "").strip()
                    ]

        except Exception as err:
            logger.warning(
                f"Structured LLM synthesis encountered an error: {err}. Attempting prompt fallback..."
            )

        # Secondary fallback: standard invoke with JSON extraction if structured output was empty
        if not output_list:
            try:
                json_prompt = (
                    f"{system_prompt}\n\n"
                    "Return a valid JSON object matching this exact schema:\n"
                    "{\n"
                    '  "articles": [\n'
                    '    {"title": "...", "summary": "...", "source_url": "...", "published_date": "..."}\n'
                    "  ],\n"
                    '  "dropped_reasoning": "..."\n'
                    "}\n"
                    f"\n{user_prompt}"
                )
                response = llm.invoke(json_prompt)
                text_content = response.content if hasattr(response, "content") else str(response)

                start_idx = text_content.find("{")
                end_idx = text_content.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = text_content[start_idx : end_idx + 1]
                    data = json.loads(json_str)
                    articles = data.get("articles", [])
                    dropped_reasoning = data.get("dropped_reasoning", "")
                    output_list = [
                        {
                            "title": a.get("title", "Untitled").strip(),
                            "summary": a.get("summary", "").strip(),
                            "source_url": a.get("source_url", "").strip(),
                            "published_date": a.get("published_date"),
                        }
                        for a in articles[:top_n]
                        if a.get("title", "").strip() and a.get("source_url", "").strip()
                    ]
            except Exception as fallback_err:
                logger.error(f"Fallback synthesis also failed: {fallback_err}")

    # If LLM synthesis succeeded, log accounting and return output_list
    if output_list:
        # Guarantee minimum 5 distinct stories if num_raw >= 5 by backfilling if the LLM produced fewer than 5
        seen_urls = {art["source_url"] for art in output_list}
        if len(output_list) < min(5, num_raw):
            for raw_item in raw_results:
                raw_url = raw_item.get("url", "").strip()
                if raw_url and raw_url not in seen_urls:
                    seen_urls.add(raw_url)
                    output_list.append({
                        "title": raw_item.get("title", "Untitled").strip(),
                        "summary": raw_item.get("snippet", "")[:350].strip() or "Key technological update.",
                        "source_url": raw_url,
                        "published_date": raw_item.get("published_date"),
                    })
                if len(output_list) >= min(5, num_raw):
                    break

        dropped_count = max(0, num_raw - len(output_list))
        if not dropped_reasoning:
            dropped_reasoning = (
                f"{dropped_count} items excluded (deduplicated overlapping coverage or filtered lower-impact items "
                f"to highlight the top {len(output_list)} distinct stories)."
            )

        logger.info(
            f"Research synthesis accounting: {num_raw} raw results in -> {len(output_list)} summaries out "
            f"({dropped_count} dropped: {dropped_reasoning})"
        )
        return output_list[:top_n]

    # Ultimate safety net: Return parsed raw items to ensure workflow never crashes
    logger.warning("Using raw search fallback formatting due to LLM synthesis failure.")
    safety_net_articles = []
    seen_urls = set()
    for item in raw_results:
        url = item.get("url", "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            safety_net_articles.append({
                "title": item.get("title", "Untitled").strip(),
                "summary": item.get("snippet", "")[:350].strip() or "Key update from industry sources.",
                "source_url": url,
                "published_date": item.get("published_date"),
            })
        if len(safety_net_articles) >= top_n:
            break

    dropped_count = max(0, num_raw - len(safety_net_articles))
    logger.info(
        f"Research synthesis accounting: {num_raw} raw results in -> {len(safety_net_articles)} summaries out "
        f"({dropped_count} dropped: LLM synthesis unavailable, formatted top {len(safety_net_articles)} distinct raw items)"
    )
    return safety_net_articles


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    print("--- Testing tools/summarizer.py in isolation ---")
    mock_raw = [
        {
            "title": "OpenAI Releases New Swarm Framework for Multi-Agent Orchestration",
            "url": "https://example.com/openai-swarm",
            "snippet": "OpenAI today published an experimental framework called Swarm for orchestrating lightweight multi-agent systems with routine patterns.",
            "published_date": "2025-04-12"
        },
        {
            "title": "Anthropic Introduces Computer Use Capability in Claude 3.5 Sonnet",
            "url": "https://example.com/claude-computer-use",
            "snippet": "Anthropic announced a major upgrade enabling Claude 3.5 Sonnet to interact directly with desktop software and browsers.",
            "published_date": "2025-04-10"
        }
    ]

    api_key = get_secret("GOOGLE_API_KEY")
    if api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model_name = get_secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
            llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
            print(f"Running synthesize_research with live Gemini model ({model_name})...")
            synthesized = synthesize_research(mock_raw, llm, top_n=2)
            for idx, a in enumerate(synthesized, 1):
                print(f"\n[{idx}] {a['title']}")
                print(f"    Source: {a['source_url']}")
                print(f"    Summary: {a['summary']}")
        except Exception as e:
            print(f"Live Gemini test skipped or encountered error: {e}")
    else:
        print("GOOGLE_API_KEY not configured. Testing raw safety net fallback:")
        synthesized = synthesize_research(mock_raw, None, top_n=2)
        print(f"Fallback returned {len(synthesized)} items successfully.")
