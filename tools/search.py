"""
tools/search.py: Tavily News Search Tool.

This module provides standalone web search functionality specialized for current news
and articles using the Tavily Search API. It operates independently of any LangGraph
state or graph structures.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Ensure workspace root is in sys.path when script is executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_secret

# Configure module logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables (e.g., TAVILY_API_KEY)
load_dotenv()


def search_news(
    queries: list[str],
    max_results_per_query: int = 5,
    days_back: int = 30
) -> list[dict]:
    """
    Executes multiple news search queries using the Tavily Search API,
    deduplicates the results by URL, and extracts key article metadata.

    Args:
        queries: A list of search query strings to execute.
        max_results_per_query: Maximum number of search results to fetch per query.
        days_back: Number of days back to filter news articles for recency (default: 30).

    Returns:
        list[dict]: A deduplicated list of search result dictionaries, each containing:
            - "title": str (Article or page title)
            - "url": str (Canonical source URL)
            - "snippet": str (Content snippet or summary text)
            - "published_date": Optional[str] (Publication date string, if available)
    """
    if not queries:
        logger.warning("search_news called with an empty queries list.")
        return []

    api_key = get_secret("TAVILY_API_KEY")
    if not api_key or api_key.startswith("your_") or api_key == "dummy_key_for_offline_validation":
        error_msg = (
            "\n" + "=" * 74 + "\n"
            "🚨 CRITICAL CONFIGURATION ERROR: TAVILY_API_KEY is missing or invalid!\n"
            "   search_news() cannot execute web searches without a configured Tavily key.\n"
            "   Returning empty results would cause silent failures or hollow newsletters.\n\n"
            "   👉 Action Required:\n"
            "      1. Get a free API key at: https://tavily.com\n"
            "      2. Set it in your .env file: TAVILY_API_KEY=tvly-your_key_here\n"
            "      3. Verify your configuration: python scripts/check_setup.py\n"
            + "=" * 74
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
    except ImportError:
        logger.error(
            "tavily-python package is not installed. "
            "Please run: pip install tavily-python"
        )
        raise ImportError("tavily-python package is required. Run: pip install tavily-python")
    except Exception as e:
        logger.error(f"Failed to initialize TavilyClient ({type(e).__name__}): {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize TavilyClient: {e}") from e

    all_results: list[dict] = []
    seen_urls: set[str] = set()

    for idx, query in enumerate(queries, 1):
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            logger.warning(f"Query #{idx} is empty or whitespace-only; skipping.")
            continue

        # Prepare parameters
        search_params = {
            "query": cleaned_query,
            "topic": "news",
            "max_results": max_results_per_query,
            "include_raw_content": False,
        }
        if days_back > 0:
            search_params["days"] = days_back

        # Explicit logging right before the API call
        logger.info(
            f"--> [Query {idx}/{len(queries)}] Dispatching Tavily API search:\n"
            f"    Query:       '{cleaned_query}'\n"
            f"    Topic:       {search_params.get('topic')}\n"
            f"    Days Back:   {search_params.get('days', 'N/A')}\n"
            f"    Max Results: {search_params.get('max_results')}"
        )

        try:
            response = client.search(**search_params)
            raw_items = response.get("results", []) if isinstance(response, dict) else []

            # Explicit logging right after the API call showing raw count
            logger.info(
                f"<-- [Query {idx}/{len(queries)}] Received {len(raw_items)} raw results "
                f"from Tavily for query: '{cleaned_query}'"
            )

            # Warning if a specific query returned 0 items
            if len(raw_items) == 0:
                logger.warning(
                    f"⚠️ Query '{cleaned_query}' yielded 0 results from Tavily "
                    f"(topic='news', days={days_back}). Query may be too narrow or out of date range."
                )

            query_added_count = 0
            for item in raw_items:
                url = item.get("url", "").strip()
                if not url:
                    continue
                if url in seen_urls:
                    logger.debug(f"Skipping duplicate URL: {url}")
                    continue

                seen_urls.add(url)
                title = item.get("title", "Untitled").strip()
                snippet = item.get("content", "").strip() or item.get("snippet", "").strip()
                published_date = item.get("published_date") or None

                all_results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "published_date": str(published_date) if published_date else None,
                })
                query_added_count += 1

            logger.info(
                f"    Added {query_added_count} new unique articles from query #{idx} "
                f"(cumulative unique: {len(all_results)})"
            )

        except Exception as err:
            err_msg = str(err).lower()
            if "unauthorized" in err_msg or "invalid" in err_msg or "401" in err_msg or "api key" in err_msg:
                auth_error = (
                    "\n" + "=" * 74 + "\n"
                    "🚨 CRITICAL AUTHENTICATION ERROR: Tavily rejected the configured API key!\n"
                    f"   Details: {err}\n\n"
                    "   👉 Action Required:\n"
                    "      1. Verify your key at: https://tavily.com\n"
                    "      2. Update your .env file with the active key: TAVILY_API_KEY=tvly-...\n"
                    "      3. Verify your configuration: python scripts/check_setup.py\n"
                    + "=" * 74
                )
                logger.error(auth_error, exc_info=True)
                raise RuntimeError(auth_error) from err

            # Detailed exception logging with error type and full message for other non-fatal errors
            logger.error(
                f"❌ Tavily API search failed for query '{cleaned_query}'!\n"
                f"    Error Type:    {type(err).__name__}\n"
                f"    Error Message: {err}",
                exc_info=True
            )
            continue

    logger.info(f"Total unique news articles accumulated across all queries: {len(all_results)}")
    return all_results


if __name__ == "__main__":
    print("--- Testing tools/search.py diagnostics in isolation ---")
    test_queries = [
        "AI agent frameworks launch 2025",
        "autonomous AI agent startup funding",
        "extremely_unlikely_random_query_that_should_yield_zero_results_12345"
    ]
    results = search_news(test_queries, max_results_per_query=3, days_back=30)
    print(f"\nFinal unique results fetched ({len(results)} items):")
    for idx, article in enumerate(results, 1):
        print(f"\n[{idx}] {article['title']}")
        print(f"    URL:  {article['url']}")
        print(f"    Date: {article['published_date']}")
        print(f"    Snippet: {article['snippet'][:120]}...")
