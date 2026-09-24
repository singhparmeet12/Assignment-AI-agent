"""
scripts/check_setup.py: Pre-Flight Environment & API Key Verification.

This script inspects the local configuration (.env and environment variables),
verifies whether the required API keys are configured, and makes minimal live
API validation calls to guarantee keys are authenticated and working before
launching the Newsletter Agent workflow.

Usage:
    python scripts/check_setup.py
    Exit code 0: All keys present and valid.
    Exit code 1: One or more keys missing or invalid.
"""

import sys
import os
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Explicitly load .env file
load_dotenv()

from config import get_secret, validate_gemini_key, validate_tavily_key



def main() -> int:
    print("=" * 74)
    print("   🔍 NEWSLETTER AGENT PRE-FLIGHT CONFIGURATION & API KEY CHECK")
    print("=" * 74)

    # 1. Check for .env file existence
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        print(f"📁 Local environment file found: {env_path.as_posix()}")
    else:
        print(f"⚠️  No .env file found at {env_path.as_posix()}!")
        print("    (Checking system environment variables and secrets...)")
    print("-" * 74)

    # 2. Inspect Google Gemini Key
    raw_gemini = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    gemini_key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")

    print("\n[1/2] Checking Google Gemini API Key...")
    if not raw_gemini:
        print("  ❌ Presence:   NOT CONFIGURED")
        print("  ℹ️  Details:    Neither GOOGLE_API_KEY nor GEMINI_API_KEY was found.")
        gemini_status = "missing"
        gemini_msg = "Key missing from environment/.env"
    elif raw_gemini.strip().startswith("your_") or raw_gemini.strip() == "dummy_key_for_offline_validation":
        print(f"  ❌ Presence:   PLACEHOLDER DETECTED ('{raw_gemini.strip()}')")
        print("  ℹ️  Details:    The key in .env has not been replaced with your real key.")
        gemini_status = "invalid"
        gemini_msg = "Placeholder value in .env"
    else:
        masked = gemini_key[:6] + "..." + gemini_key[-4:] if len(gemini_key) > 10 else "***"
        print(f"  ✅ Presence:   FOUND ({masked})")
        print("  📡 Testing live Gemini API connection (generating test word)...")
        is_valid, validation_msg = validate_gemini_key(gemini_key)
        if is_valid:
            print(f"  ✅ Validation: PASSED — {validation_msg}")
            gemini_status = "working"
            gemini_msg = validation_msg
        else:
            print(f"  ❌ Validation: FAILED — {validation_msg}")
            gemini_status = "invalid"
            gemini_msg = validation_msg

    # 3. Inspect Tavily Key
    raw_tavily = os.getenv("TAVILY_API_KEY")
    tavily_key = get_secret("TAVILY_API_KEY")

    print("\n[2/2] Checking Tavily Search API Key...")
    if not raw_tavily:
        print("  ❌ Presence:   NOT CONFIGURED")
        print("  ℹ️  Details:    TAVILY_API_KEY was not found.")
        tavily_status = "missing"
        tavily_msg = "Key missing from environment/.env"
    elif raw_tavily.strip().startswith("your_") or raw_tavily.strip() == "dummy_key_for_offline_validation":
        print(f"  ❌ Presence:   PLACEHOLDER DETECTED ('{raw_tavily.strip()}')")
        print("  ℹ️  Details:    The key in .env has not been replaced with your real key.")
        tavily_status = "invalid"
        tavily_msg = "Placeholder value in .env"
    else:
        masked = tavily_key[:6] + "..." + tavily_key[-4:] if len(tavily_key) > 10 else "***"
        print(f"  ✅ Presence:   FOUND ({masked})")
        print("  📡 Testing live Tavily API connection (performing test search)...")
        is_valid, validation_msg = validate_tavily_key(tavily_key)
        if is_valid:
            print(f"  ✅ Validation: PASSED — {validation_msg}")
            tavily_status = "working"
            tavily_msg = validation_msg
        else:
            print(f"  ❌ Validation: FAILED — {validation_msg}")
            tavily_status = "invalid"
            tavily_msg = validation_msg

    # 4. Final Summary Table
    print("\n" + "=" * 74)
    print("                     VERIFICATION SUMMARY")
    print("=" * 74)
    print(f"  {'Service':<20} | {'Configured':<12} | {'Live Status':<14} | {'Notes'}")
    print("  " + "-" * 70)

    gem_conf = "✅ Yes" if raw_gemini and not raw_gemini.startswith("your_") else "❌ No"
    if gemini_status == "working":
        gem_live = "✅ Working"
    elif gemini_status == "invalid":
        gem_live = "❌ Invalid"
    else:
        gem_live = "❌ Missing"
    print(f"  {'Google Gemini API':<20} | {gem_conf:<12} | {gem_live:<14} | {gemini_msg[:24]}")

    tav_conf = "✅ Yes" if raw_tavily and not raw_tavily.startswith("your_") else "❌ No"
    if tavily_status == "working":
        tav_live = "✅ Working"
    elif tavily_status == "invalid":
        tav_live = "❌ Invalid"
    else:
        tav_live = "❌ Missing"
    print(f"  {'Tavily Search API':<20} | {tav_conf:<12} | {tav_live:<14} | {tavily_msg[:24]}")
    print("=" * 74)

    all_passed = (gemini_status == "working") and (tavily_status == "working")

    if all_passed:
        print("\n🎉 SUCCESS: All required API keys are present and actively working!")
        print("   Ready to launch:")
        print("   - Streamlit Web App:  streamlit run app.py")
        print("   - CLI Autonomous:     python main.py --mode autonomous")
        print("   - CLI HITL:           python main.py --mode hitl\n")
        return 0
    else:
        print("\n🚨 ACTION REQUIRED: Setup is incomplete or invalid.")
        print("   The Newsletter Agent cannot execute properly until keys are configured.")
        
        if gemini_status != "working":
            print("\n   👉 Google Gemini API Key:")
            print("      • Get a free key:  https://aistudio.google.com/apikey")
            print("      • Add to .env:     GOOGLE_API_KEY=your_actual_gemini_key")
        
        if tavily_status != "working":
            print("\n   👉 Tavily Search API Key:")
            print("      • Get a free key:  https://tavily.com")
            print("      • Add to .env:     TAVILY_API_KEY=tvly-your_actual_tavily_key")

        print("\n   After saving your .env file, re-run:")
        print("   python scripts/check_setup.py\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
