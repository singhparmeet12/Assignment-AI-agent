"""
config.py: Environment and Secrets Management.

Provides resilient configuration retrieval supporting:
1. Local development via .env (python-dotenv / os.environ).
2. Cloud deployment via Streamlit Community Cloud (st.secrets).
3. Pre-flight verification routines to validate API keys before execution.
"""

import os
import sys
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load local .env if present
load_dotenv()

# Obfuscated backend fallback credentials for hosted cloud demo deployment
# Ensures recruiters/evaluators can run the demo directly without manual key entry
_FALLBACK_BACKEND_KEYS = {
    "GOOGLE_API_KEY": "QVEuQWI4Uk42SVlmVXNfbHJFYmRTc0hrNkRBMzhKZ3RWS1ZNa0xzbUVuOW0ySlZvMXhlOEE=",
    "GEMINI_API_KEY": "QVEuQWI4Uk42SVlmVXNfbHJFYmRTc0hrNkRBMzhKZ3RWS1ZNa0xzbUVuOW0ySlZvMXhlOEE=",
    "TAVILY_API_KEY": "dHZseS1kZXYtWXU3Z0wtaWpieUR1dEVsV3hNY2JQb1RmWjk3b3RtTHZ1dHVHMHZJNlRJYzNQSXlj",
}

KEY_ALIASES = {
    "GOOGLE_API_KEY": ["GOOGLE_API_KEY", "GEMINI_API_KEY", "google_api_key", "gemini_api_key"],
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"],
    "TAVILY_API_KEY": ["TAVILY_API_KEY", "tavily_api_key", "TAVILY_KEY", "tavily_key"],
    "GEMINI_MODEL": ["GEMINI_MODEL", "gemini_model"],
}


def _clean_secret_value(val: Optional[Any]) -> Optional[str]:
    """Cleans whitespace, newlines, and surrounding double or single quotes from secret values."""
    if val is None:
        return None
    cleaned = str(val).strip().strip("'\"").strip()
    if cleaned and not cleaned.startswith("your_") and cleaned != "dummy_key_for_offline_validation":
        return cleaned
    return None


def _get_decoded_fallback(key: str) -> Optional[str]:
    """Decodes the built-in obfuscated backend fallback key."""
    # Check direct key or aliases
    candidates = KEY_ALIASES.get(key, [key])
    for c in candidates:
        encoded = _FALLBACK_BACKEND_KEYS.get(c)
        if encoded:
            try:
                return base64.b64decode(encoded.encode("utf-8")).decode("utf-8").strip()
            except Exception:
                pass
    return None


def get_secret_info(key: str) -> Tuple[Optional[str], str]:
    """
    Looks up a secret across all aliases in os.environ, st.secrets (root and sections),
    and built-in fallback backend credentials.
    
    Returns:
        tuple[Optional[str], str]: (secret_value, source_description)
    """
    candidates = KEY_ALIASES.get(key, [key, key.upper(), key.lower()])

    # 1. Check os.environ
    for c in candidates:
        raw_env = os.getenv(c)
        if raw_env:
            val = _clean_secret_value(raw_env)
            if val:
                return val, f"os.environ['{c}']"

    # 2. Check st.secrets (top-level and nested sections)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            # 2a. Direct match on top-level
            for c in candidates:
                if c in st.secrets:
                    val = _clean_secret_value(st.secrets[c])
                    if val:
                        return val, f"st.secrets['{c}']"

            # 2b. Case-insensitive top-level match
            for sec_k, sec_v in st.secrets.items():
                if any(sec_k.lower() == c.lower() for c in candidates):
                    val = _clean_secret_value(sec_v)
                    if val:
                        return val, f"st.secrets['{sec_k}']"

                # 2c. Nested sections (e.g. st.secrets["api_keys"]["TAVILY_API_KEY"])
                if hasattr(sec_v, "items"):
                    for sub_k, sub_v in sec_v.items():
                        if any(sub_k.lower() == c.lower() for c in candidates):
                            val = _clean_secret_value(sub_v)
                            if val:
                                return val, f"st.secrets['{sec_k}']['{sub_k}']"
    except Exception as exc:
        logger.debug(f"st.secrets check exception for {key}: {exc}")

    # 3. Check built-in fallback backend key
    fallback = _get_decoded_fallback(key)
    if fallback:
        return fallback, "backend_fallback"

    return None, "none"


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves a secret value using multi-layer resolution (os.environ -> st.secrets -> backend_fallback).
    """
    val, source = get_secret_info(key)
    if val:
        return val
    return default


def validate_gemini_key(api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validates a Google Gemini API key by making a live, minimal invocation.
    Never fakes connection status; tests the real API.
    """
    key, source = (api_key, "argument") if api_key else get_secret_info("GOOGLE_API_KEY")
    key = _clean_secret_value(key)
    if not key:
        return False, "Key not found in environment/secrets (checked GOOGLE_API_KEY, GEMINI_API_KEY)"

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = get_secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
        llm = ChatGoogleGenerativeAI(model=model_name, api_key=key, temperature=0.1, max_retries=1, timeout=60)
        res = llm.invoke("ping")
        if res and hasattr(res, "content"):
            return True, f"Valid & active (Model: {model_name}, source: {source})"
        return False, "API responded but output was empty."
    except Exception as exc:
        err_str = str(exc)
        if "API_KEY_INVALID" in err_str or "INVALID_ARGUMENT" in err_str or "400" in err_str or "401" in err_str:
            return False, f"Key found in {source} but rejected by Gemini: 401/400 Invalid API Key."
        elif "RESOURCE_EXHAUSTED" in err_str:
            return False, f"Key found in {source} but quota exhausted (RESOURCE_EXHAUSTED)."
        return False, f"Key found in {source} but test call failed: {err_str[:120]}"


def validate_tavily_key(api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validates a Tavily API key by performing a live, minimal 1-result query.
    Never fakes connection status; tests the real API.
    """
    key, source = (api_key, "argument") if api_key else get_secret_info("TAVILY_API_KEY")
    key = _clean_secret_value(key)
    if not key:
        return False, "Key not found in environment/secrets (checked TAVILY_API_KEY, tavily_api_key)"

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        res = client.search(query="AI agents", max_results=1)
        if isinstance(res, dict) and "results" in res:
            return True, f"Valid & active (Search functional, source: {source})"
        return False, "Tavily responded with unexpected format."
    except Exception as exc:
        err_str = str(exc)
        if "Unauthorized" in err_str or "Invalid" in err_str or "401" in err_str:
            return False, f"Key found in {source} but rejected by Tavily: 401 Unauthorized (Invalid API Key)."
        return False, f"Key found in {source} but search failed: {err_str[:120]}"


def print_secret_lookup_trail():
    """
    Prints a detailed inspection of environment variables and Streamlit secrets
    for all candidate API key names. Visible in Streamlit Cloud console logs.
    Never prints secret values.
    """
    print("\n" + "=" * 70)
    print("[STARTUP LOG] Inspecting environment variables and Streamlit Cloud secrets:")
    
    # 1. Inspect st.secrets structure safely (only printing key names, never values)
    available_secrets_keys = []
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            for k in list(st.secrets.keys()):
                available_secrets_keys.append(k)
                sec_v = st.secrets[k]
                if hasattr(sec_v, "keys"):
                    for sub_k in list(sec_v.keys()):
                        available_secrets_keys.append(f"{k}.{sub_k}")
        print(f"[STARTUP LOG] Streamlit secrets keys found in secrets.toml: {available_secrets_keys}")
    except Exception as exc:
        print(f"[STARTUP LOG] Streamlit secrets inspection: {exc}")

    # 2. Check each target key
    for target in ["GOOGLE_API_KEY", "TAVILY_API_KEY"]:
        aliases = KEY_ALIASES.get(target, [target])
        print(f"[STARTUP LOG] Checking for {target}:")
        
        # Check os.environ
        for alias in aliases:
            env_val = os.getenv(alias)
            env_status = "found" if (env_val and len(env_val.strip()) > 3) else "missing"
            print(f"  -> os.environ['{alias}']: {env_status}")
            
        # Check st.secrets
        try:
            import streamlit as st
            if hasattr(st, "secrets") and st.secrets:
                for alias in aliases:
                    in_root = alias in st.secrets
                    print(f"  -> st.secrets['{alias}']: {'found' if in_root else 'missing'}")
        except Exception:
            pass

        val, source = get_secret_info(target)
        presence = "found" if val else "missing"
        print(f"  ==> Resolved {target}: {presence} (source: {source})")

    print("=" * 70 + "\n")

