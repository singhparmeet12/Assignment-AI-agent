"""
config.py: Environment and Secrets Management.

Provides resilient configuration retrieval supporting:
1. Local development via .env (python-dotenv / os.environ).
2. Cloud deployment via Streamlit Community Cloud (st.secrets).
3. Pre-flight verification routines to validate API keys before execution.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves a secret or configuration value.
    Checks os.environ first (local .env), then falls back to st.secrets
    if running in Streamlit Community Cloud.

    Args:
        key: The environment or secret key name.
        default: Optional fallback value if the key is not found.

    Returns:
        Optional[str]: The secret value or default.
    """
    # 1. Check local environment variables (.env / OS)
    val = os.getenv(key)
    if val and val.strip() and not val.strip().startswith("your_"):
        return val.strip()

    # 2. Check Streamlit Community Cloud secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            secret_val = str(st.secrets[key]).strip()
            if secret_val and not secret_val.startswith("your_"):
                return secret_val
    except Exception:
        # st.secrets is unavailable outside Streamlit context
        pass

    return default


def validate_gemini_key(api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validates a Google Gemini API key by making a minimal invocation.

    Returns:
        tuple[bool, str]: (is_valid, status_or_error_message)
    """
    key = api_key or get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
    if not key:
        return False, "Missing: No key configured. Get one at https://aistudio.google.com/apikey"
    if key.startswith("your_") or key == "dummy_key_for_offline_validation":
        return False, "Placeholder key detected. Replace with a real key from https://aistudio.google.com/apikey"

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = get_secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
        llm = ChatGoogleGenerativeAI(model=model_name, api_key=key, temperature=0.1, max_retries=2, timeout=15)
        res = llm.invoke("ping")
        if res and hasattr(res, "content"):
            return True, f"Valid & active (Model: {model_name})"
        return False, "API responded but output was empty."
    except Exception as exc:
        err_str = str(exc)
        if "API_KEY_INVALID" in err_str or "INVALID_ARGUMENT" in err_str:
            return False, "Authentication failed: Key rejected by Google Gemini (INVALID_ARGUMENT)."
        elif "RESOURCE_EXHAUSTED" in err_str:
            return False, "Quota exhausted (RESOURCE_EXHAUSTED): Rate limit reached."
        return False, f"Connection/API error: {err_str[:120]}"


def validate_tavily_key(api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validates a Tavily API key by performing a minimal 1-result query.

    Returns:
        tuple[bool, str]: (is_valid, status_or_error_message)
    """
    key = api_key or get_secret("TAVILY_API_KEY")
    if not key:
        return False, "Missing: No key configured. Get one at https://tavily.com"
    if key.startswith("your_"):
        return False, "Placeholder key detected. Replace with a real key from https://tavily.com"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        res = client.search(query="AI agents", max_results=1)
        if isinstance(res, dict) and "results" in res:
            return True, "Valid & active (Search functional)"
        return False, "Tavily responded with unexpected format."
    except Exception as exc:
        err_str = str(exc)
        if "Unauthorized" in err_str or "Invalid" in err_str or "401" in err_str:
            return False, "Authentication failed: Key rejected by Tavily (Unauthorized/Invalid)."
        return False, f"Tavily API error: {err_str[:120]}"
