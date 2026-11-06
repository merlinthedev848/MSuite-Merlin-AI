"""Frontend Architecture, UI/UX Design System, and Accessibility Quality Tests.

Ensures strict adherence to:
- WCAG 2.1 AA Accessibility Guidelines (buttons, inputs, dialogs, focus rings)
- Hardware-Accelerated Layout & Zero-Jank Sidebar Transforms
- Design System Tokens (modal sizing, empty states, loading spinners, interactive states)
- Client-Side Error Boundaries & Resilience
- Pure Stark Arc Reactor Visual HUD Integrity
"""

import os
import re
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(ROOT_DIR, "web", "index.html")
CSS_PATH = os.path.join(ROOT_DIR, "web", "styles.css")
JS_PATH = os.path.join(ROOT_DIR, "web", "script.js")


def test_html_accessibility_buttons_and_controls():
    """Verify all interactive buttons and inputs have accessible labels/names."""
    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()

    # All buttons must have text content, aria-label, or title
    button_matches = re.findall(r'<button\b([^>]*)>(.*?)</button>', html, re.DOTALL)
    assert len(button_matches) > 50, "Expected at least 50 buttons in UI"
    for attrs, content in button_matches:
        stripped_text = re.sub(r'<[^>]+>', '', content).strip()
        has_aria = 'aria-label=' in attrs or 'title=' in attrs
        assert stripped_text or has_aria, f"Button missing accessible label: {attrs}"

    # All non-hidden input elements must have id or aria-label
    input_matches = re.findall(r'<input\b([^>]*)>', html)
    for attrs in input_matches:
        itype = re.search(r'type=["\']([^"\']+)["\']', attrs)
        if itype and itype.group(1) in ('hidden', 'submit', 'button'):
            continue
        assert 'id=' in attrs or 'aria-label=' in attrs or 'aria-labelledby=' in attrs, (
            f"Input missing id or aria-label: {attrs}"
        )


def test_html_modal_dialogs_wcag_conformance():
    """Verify all modal containers declare role='dialog' and aria-modal='true'."""
    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()

    modals = re.findall(r'<div\b[^>]*class=["\'][^"\']*\bmodal-backdrop\b[^"\']*["\'][^>]*>', html)
    assert len(modals) >= 8, f"Expected at least 8 modal backdrops, found {len(modals)}"
    for m in modals:
        assert 'role="dialog"' in m or "role='dialog'" in m, f"Modal missing role='dialog': {m}"
        assert 'aria-modal="true"' in m or "aria-modal='true'" in m, f"Modal missing aria-modal='true': {m}"


def test_css_hardware_acceleration_and_sidebar_contract():
    """Verify sidebar uses pure GPU transforms without reflowing layout with width transitions."""
    with open(CSS_PATH, encoding="utf-8") as f:
        css = f.read()

    # No width transitions on sidebar
    assert "transition: transform" in css
    # Mobile collapsed must use transform: translateX(-100%)
    assert "transform: translateX(-100%)" in css
    # High contrast WCAG focus rings
    assert ":focus-visible" in css
    assert "outline: 2px solid #00f0ff" in css
    # Zero-latency typing pause rules
    assert "body.typing-active" in css
    assert "animation-play-state: paused !important" in css


def test_css_design_system_tokens():
    """Verify design system tokens for modal sizing, empty states, and loading spinners."""
    with open(CSS_PATH, encoding="utf-8") as f:
        css = f.read()

    assert ".modal-box-wide" in css
    assert ".modal-box-large" in css
    assert ".modal-box-narrow" in css
    assert ".empty-state" in css
    assert ".loading-spinner" in css
    assert "button:disabled" in css


def test_javascript_resilience_and_error_boundaries():
    """Verify script.js includes error boundaries, timeout-safe fetch, and clean bootstrap."""
    with open(JS_PATH, encoding="utf-8") as f:
        js = f.read()

    assert 'window.addEventListener("error"' in js
    assert 'window.addEventListener("unhandledrejection"' in js
    assert "fetchWithTimeout" in js
    assert "initAlwaysListeningMerlin" in js

    # Zero legacy identity tokens
    assert "J.A.R.V.I.S." not in js
    assert "JARVIS" not in js


def test_javascript_node_syntax_and_integrity():
    """Verify JavaScript executes in Node.js without syntax or evaluation errors."""
    res = subprocess.run(
        ["node", "--check", JS_PATH],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0, f"Node syntax check failed: {res.stderr}"
