"""
tools/html_generator.py: Markdown to Responsive Email HTML Converter.

This module converts raw markdown newsletter drafts into responsive,
email-client-compatible HTML email templates with inline styles suitable
for Gmail, Apple Mail, Outlook, and webmail clients.
It operates independently of LangGraph state structures.
"""

import re
import datetime
from typing import Optional

try:
    import markdown
except ImportError:
    markdown = None


def markdown_to_email_html(
    markdown_text: str,
    subject_line: str,
    preheader: Optional[str] = None
) -> str:
    """
    Converts a markdown newsletter body into a responsive, inline-styled
    HTML email suitable for desktop and mobile email clients.

    Args:
        markdown_text: The complete markdown content of the newsletter.
        subject_line: The subject line to display prominently in the header.
        preheader: Optional short preview snippet displayed in email inbox previews.

    Returns:
        str: Self-contained, email-safe HTML string with inline CSS.
    """
    # Fallback if markdown library is missing
    if markdown is not None:
        raw_body_html = markdown.markdown(
            markdown_text,
            extensions=["extra", "sane_lists", "nl2br"]
        )
    else:
        # Basic conversion fallback
        paragraphs = markdown_text.strip().split("\n\n")
        raw_body_html = "".join(f"<p>{p.replace('\n', '<br>')}</p>" for p in paragraphs)

    # Post-process common HTML tags with inline styles for cross-client email fidelity
    body_styled = _apply_email_inline_styles(raw_body_html)

    # Generate current date for header badge
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    preview_snippet = preheader or (subject_line[:120] if subject_line else "Your curated AI briefing.")

    full_html = f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>{subject_line}</title>
    <!--[if mso]>
    <noscript>
        <xml>
            <o:OfficeDocumentSettings>
                <o:PixelsPerInch>96</o:PixelsPerInch>
            </o:OfficeDocumentSettings>
        </xml>
    </noscript>
    <![endif]-->
</head>
<body style="margin: 0; padding: 0; background-color: #f4f5f7; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; -webkit-font-smoothing: antialiased; color: #1e293b; line-height: 1.6;">

    <!-- Hidden Preheader Text for Inbox Preview -->
    <div style="display: none; max-height: 0px; overflow: hidden; font-size: 1px; line-height: 1px; color: #fff; opacity: 0;">
        {preview_snippet}
    </div>

    <!-- Outer Centering Table -->
    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f4f5f7; width: 100%; border-collapse: collapse;">
        <tr>
            <td align="center" style="padding: 28px 12px 40px 12px;">
                
                <!-- Main Container Card (640px max width) -->
                <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 640px; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06); border: 1px solid #e2e8f0; border-collapse: separate;">
                    
                    <!-- Header Section -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%); padding: 36px 32px 30px 32px; text-align: left; color: #ffffff;">
                            
                            <!-- Issue Meta Badge -->
                            <table role="presentation" border="0" cellpadding="0" cellspacing="0" style="margin-bottom: 14px;">
                                <tr>
                                    <td style="background: rgba(255, 255, 255, 0.15); border: 1px solid rgba(255, 255, 255, 0.25); border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: #e0e7ff;">
                                        THE AGENTIC DISPATCH &bull; {current_date}
                                    </td>
                                </tr>
                            </table>

                            <!-- Subject Title -->
                            <h1 style="margin: 0 0 10px 0; font-size: 26px; line-height: 1.3; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">
                                {subject_line}
                            </h1>

                            <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #c7d2fe;">
                                Curated insights, technical breakthroughs, and practical intelligence in autonomous AI.
                            </p>
                        </td>
                    </tr>

                    <!-- Decorative Accent Strip -->
                    <tr>
                        <td height="4" style="background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%); line-height: 4px; font-size: 4px;">&nbsp;</td>
                    </tr>

                    <!-- Newsletter Body -->
                    <tr>
                        <td style="padding: 36px 32px 28px 32px; font-size: 15px; color: #334155; line-height: 1.7;">
                            {body_styled}
                        </td>
                    </tr>

                    <!-- Divider -->
                    <tr>
                        <td style="padding: 0 32px;">
                            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 0;">
                        </td>
                    </tr>

                    <!-- Footer Section -->
                    <tr>
                        <td style="padding: 28px 32px 32px 32px; background-color: #f8fafc; text-align: center; color: #64748b; font-size: 13px; line-height: 1.6;">
                            <p style="margin: 0 0 8px 0; font-weight: 600; color: #475569;">
                                The Agentic Dispatch &bull; Automated Intelligence
                            </p>
                            <p style="margin: 0 0 16px 0;">
                                You received this newsletter because you are subscribed to automated research updates.
                            </p>
                            <p style="margin: 0; font-size: 12px; color: #94a3b8;">
                                <a href="#view-in-browser" style="color: #6366f1; text-decoration: underline; margin: 0 8px;">View in Browser</a> &bull;
                                <a href="#manage-preferences" style="color: #6366f1; text-decoration: underline; margin: 0 8px;">Preferences</a> &bull;
                                <a href="#unsubscribe" style="color: #6366f1; text-decoration: underline; margin: 0 8px;">Unsubscribe</a>
                            </p>
                        </td>
                    </tr>

                </table>
                <!-- End Main Container Card -->

            </td>
        </tr>
    </table>
</body>
</html>"""

    return full_html


def _apply_email_inline_styles(html: str) -> str:
    """Applies email-safe inline styles to standard HTML elements."""
    replacements = [
        # Headings
        (r'<h2>', r'<h2 style="font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 28px; margin-bottom: 12px; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; letter-spacing: -0.3px;">'),
        (r'<h3>', r'<h3 style="font-size: 17px; font-weight: 600; color: #1e293b; margin-top: 20px; margin-bottom: 8px;">'),
        (r'<h4>', r'<h4 style="font-size: 15px; font-weight: 600; color: #334155; margin-top: 16px; margin-bottom: 6px;">'),

        # Paragraphs & Lists
        (r'<p>', r'<p style="margin: 0 0 16px 0; line-height: 1.7; color: #334155; font-size: 15px;">'),
        (r'<ul>', r'<ul style="margin: 0 0 18px 0; padding-left: 22px; color: #334155; font-size: 15px; line-height: 1.7;">'),
        (r'<ol>', r'<ol style="margin: 0 0 18px 0; padding-left: 22px; color: #334155; font-size: 15px; line-height: 1.7;">'),
        (r'<li>', r'<li style="margin-bottom: 6px;">'),

        # Links & Strong Text
        (r'<a href=', r'<a style="color: #4f46e5; text-decoration: underline; font-weight: 500;" href='),
        (r'<strong>', r'<strong style="color: #0f172a; font-weight: 600;">'),

        # Blockquotes
        (r'<blockquote>', r'<blockquote style="margin: 16px 0 20px 0; padding: 12px 18px; border-left: 4px solid #6366f1; background-color: #f1f5f9; color: #475569; font-style: italic; border-radius: 0 8px 8px 0;">'),

        # Inline Code & Code Blocks
        (r'<code>', r'<code style="background-color: #f1f5f9; color: #0f172a; padding: 2px 6px; border-radius: 4px; font-family: Consolas, Monaco, monospace; font-size: 13px;">'),
        (r'<pre>', r'<pre style="background-color: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 8px; overflow-x: auto; font-family: Consolas, Monaco, monospace; font-size: 13px; line-height: 1.5;">'),

        # Horizontal Rules
        (r'<hr>', r'<hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;">'),
        (r'<hr />', r'<hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;">'),
    ]

    for pattern, repl in replacements:
        html = re.sub(pattern, repl, html)

    return html


if __name__ == "__main__":
    print("--- Testing tools/html_generator.py in isolation ---")
    sample_md = """
## Top Story: The Shift to Autonomous AI Teams

The artificial intelligence landscape is witnessing a generational transition from single-prompt chat interfaces to **collaborative multi-agent networks**.

### Why This Matters Now
Recent benchmarks show that structured reasoning graphs outperform traditional single-pass LLMs across complex problem sets:
- **Planning & Verification:** Separating researcher and reviewer roles reduces hallucinations by up to 40%.
- **Self-Correction Loops:** Multi-turn critique loops detect logical inconsistencies before output reaches human operators.

> "The true unit of intelligence in 2025 is not the model weights, but the verification loop orchestrating them."

Read the full analysis on [DeepLearning.ai](https://example.com/deeplearning-ai-agents).
"""
    sample_subject = "AI Dispatch #42: The Architecture of Multi-Agent Systems"
    email_html = markdown_to_email_html(sample_md, sample_subject)

    print(f"Generated HTML string length: {len(email_html)} characters.")
    print("HTML Preview (first 400 chars):")
    print(email_html[:400])
    print("\n... HTML contains unsubscribe link:", "unsubscribe" in email_html.lower())
    print("... HTML contains subject line:", sample_subject in email_html)
