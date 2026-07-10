"""
TruthLens - Fake News Detection System UI
Modern, responsive Streamlit interface for real-time news article analysis.

This UI module handles all user interaction and visualization.
Logic is delegated to truthlens_detection_system.py
Styling is managed through style.css
"""

import csv
import html
import re
import time
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Import the detection system logic
from truthlens_detection_system import (
    analyze_article,
    validate_article,
    format_confidence,
    get_signal_emoji,
    get_model_info,
)

# Page config
st.set_page_config(
    page_title="TruthLens - Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load custom CSS
def load_custom_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_custom_css()


def sanitize_article_text(text: str) -> str:
    cleaned_text = html.unescape(text)
    cleaned_text = re.sub(r'<[^>]+>', ' ', cleaned_text)
    cleaned_text = re.sub(r'&nbsp;|&amp;|&lt;|&gt;|&#160;', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


def log_prediction(article_text: str, prediction, analysis_duration_seconds: float = 0.0):
    csv_path = Path(__file__).resolve().parent.parent / "logs" / "prediction_logs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    normalized_text = re.sub(r"\s+", " ", article_text).strip()
    snippet = normalized_text[:100]
    text_length = len(normalized_text)
    model_info = get_model_info()
    model_version = model_info.get("version", "unknown")
    headers = [
        "timestamp",
        "article_text_snippet",
        "text_length",
        "prediction_label",
        "confidence_percent",
        "real_probability",
        "fake_probability",
        "model_version",
        "analysis_duration_seconds",
    ]

    try:
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        with open(csv_path, "a", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow([
                timestamp,
                snippet,
                text_length,
                prediction.label,
                f"{prediction.confidence * 100:.2f}",
                f"{prediction.real_prob:.6f}",
                f"{prediction.fake_prob:.6f}",
                model_version,
                f"{analysis_duration_seconds:.3f}",
            ])
    except Exception:
        pass

# Session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "article_input" not in st.session_state:
    st.session_state.article_input = ""
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# Navbar
def render_navbar():
    navbar_html = """
    <div class="tl-nav"><div class="tl-nav-brand">🔍 Truth<span class="accent">Lens</span></div></div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)

# Hero
def render_hero():
    hero_html = """
    <div class="tl-hero">
        <div class="tl-hero-badge">AI-Powered Detection</div>
        <h1>Is this news <span class="real">real</span> or <span class="fake">fake</span>?</h1>
        <p>Paste any news article and TruthLens will analyse its language patterns, source signals, and writing structure to estimate credibility — in seconds.</p>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

# Input
# Define helpers first
def clear_input():
    st.session_state.article_input = ""
    st.session_state.analysis_result = None
    st.session_state.error_message = None
    st.session_state.is_loading = False

def sanitize_input():
    cleaned_value = sanitize_article_text(st.session_state.article_input)
    if cleaned_value != st.session_state.article_input:
        st.session_state.article_input = cleaned_value


def render_loader():
    loader_html = """
    <div class="tl-loader">
        <div class="tl-loader-ring"></div>
        <div class="tl-loader-text">Analyzing article...</div>
    </div>
    """
    st.markdown(loader_html, unsafe_allow_html=True)


def run_analysis_with_loader(article_text):
    with st.spinner("Analyzing article..."):
        prediction = analyze_article(article_text)
        time.sleep(1.2)
    return prediction

# Input section
def render_input_section():

    # Text area
    article_text = st.text_area(
        label="Article Text",
        placeholder="Paste the full text of the news article here... The more complete the article, the more accurate the analysis will be.",
        height=260,
        label_visibility="collapsed",
        key="article_input",
        on_change=sanitize_input,
    )

    # Buttons row (still inside the card)
    col1, col2 = st.columns([1, 1], gap="small")
    with col1:
        analyze_button = st.button("🔍 Analyse Article", use_container_width=True, key="analyze_btn")
    with col2:
        st.button("✖ Clear", use_container_width=True, key="clear_btn", on_click=clear_input)

    if st.session_state.is_loading and not st.session_state.analysis_result:
        render_loader()

    return article_text, analyze_button


# Results
def render_verdict(prediction, animate: bool = False):
    is_real = prediction.label == "REAL"
    verdict_class = "tl-result-real" if is_real else "tl-result-fake"
    verdict_label = prediction.label
    verdict_icon = "🛡️"
    real_pct = prediction.real_prob * 100
    fake_pct = prediction.fake_prob * 100
    animate_class = " tl-roll" if animate else ""

    badge_class = 'real' if is_real else 'fake'
    verdict_html = (
        f"<div id='t1-result-card' class='tl-result-card{animate_class} {verdict_class}'>"
        f"<div class='tl-result-header'>ANALYSIS RESULT</div>"
        f"<div class='tl-result-main'>"
        f"<div class='tl-result-summary'><div class='tl-result-badge {badge_class}'>{verdict_icon} {verdict_label}</div></div>"
        f"<div class='tl-confidence'><div class='tl-confidence-label'>Model confidence</div>"
        f"<div class='tl-confidence-value'>{format_confidence(prediction.confidence)}</div></div>"
        f"</div>"
        f"<div class='tl-bar-labels'><span class='real-label'>Real - {format_confidence(prediction.real_prob)}</span>"
        f"<span class='fake-label'>Fake - {format_confidence(prediction.fake_prob)}</span></div>"
        f"<div class='tl-bar-track'><div class='tl-bar-fill-real' style='width: {real_pct:.1f}%;'></div>"
        f"<div class='tl-bar-fill-fake' style='width: {fake_pct:.1f}%;'></div></div>"
        f"</div>"
    )
    st.markdown(verdict_html, unsafe_allow_html=True)


def render_signals(signals):
    signal_cards = []
    for signal in signals:
        dot_class = 'pos' if signal.positive else 'neg'
        signal_cards.append(
            f"<div class='tl-signal'><div class='tl-signal-header'><span class='tl-signal-dot {dot_class}'></span>"
            f"<div><div class='tl-signal-label'>{signal.label}</div><div class='tl-signal-value'>{signal.value}</div></div></div>"
            f"<div class='tl-signal-note'>{signal.note}</div></div>"
        )

    signals_html = """
    <div class="tl-signal-container">
        <div class="tl-card-label">Detection Signals</div>
        <div class="tl-signals">
            {cards}
        </div>
    </div>
    """.format(cards="".join(signal_cards))
    st.markdown(signals_html, unsafe_allow_html=True)


def scroll_to_results():
    components.html(
        """
        <script>
        function scrollToResult() {
            const parentDoc = window.parent.document;
            const result = parentDoc.getElementById('t1-result-card');
            const container = parentDoc.querySelector('[data-testid="stMain"]');
            if (result && container) {
                const rect = result.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();
                const offset = rect.top - containerRect.top - 24;
                container.scrollTo({ top: container.scrollTop + offset, behavior: 'smooth' });
            } else {
                window.setTimeout(scrollToResult, 180);
            }
        }
        window.setTimeout(scrollToResult, 260);
        </script>
        """,
        height=0,
        scrolling=False,
    )


def render_results(prediction, animate: bool = False):
    render_verdict(prediction, animate=animate)
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    render_signals(prediction.signals)
    scroll_to_results()

# Alerts
def render_error(message: str):
    error_html = f"""
    <div class="tl-alert tl-alert-error">
        <span style="font-size: 1.2rem;">⚠️</span>
        <div>
            <strong>Error:</strong> {message}
        </div>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)


def render_info(message: str):
    info_html = f"""
    <div class="tl-alert tl-alert-info">
        <span style="font-size: 1.2rem;">ℹ️</span>
        <div>{message}</div>
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)

# Information Section (updated)
def render_info_section():
    info_html = """<div class="tl-info">
<h3 style="letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: unset; padding: 0px;">HOW IT WORKS</h3>
<p style="margin-bottom: 0.75rem; color: var(--text-muted);">TruthLens analyses the linguistic and structural features of a news article using a machine-learning model trained on thousands of labelled samples. It detects patterns statistically associated with misleading content — it does not fact-check specific claims.</p>
<div class="tl-features">
    <div class="tl-feature">
        <div class="tl-feature-icon">📝</div>
        <div class="tl-feature-title">Language Analysis</div>
        <div class="tl-feature-body">Detects sensational phrasing, emotional manipulation, and unusual capitalisation patterns.</div>
    </div>
    <div class="tl-feature">
        <div class="tl-feature-icon">🔗</div>
        <div class="tl-feature-title">Source Signals</div>
        <div class="tl-feature-body">Looks for references to studies, officials, institutions, and named credible sources.</div>
    </div>
    <div class="tl-feature">
        <div class="tl-feature-icon">📊</div>
        <div class="tl-feature-title">Structural Features</div>
        <div class="tl-feature-body">Measures article length, sentence complexity, and overall writing structure.</div>
    </div>
</div>
</div>"""
    st.markdown(info_html, unsafe_allow_html=True)
    
    st.markdown("---")
    footer_note = '''<div style="text-align:center; color:var(--text-muted); font-size:0.9rem; padding:0.8rem 0;">TruthLens &middot; Fake News Detection &middot; For educational and research use only</div>'''
    st.markdown(footer_note, unsafe_allow_html=True)

# Main
def main():
    render_navbar()
    render_hero()
    article_text, analyze_button = render_input_section()

    if analyze_button:
        cleaned_text = sanitize_article_text(article_text)
        if cleaned_text != article_text:
            st.session_state.article_input = cleaned_text
            article_text = cleaned_text

        is_valid, error_msg = validate_article(article_text)
        if not is_valid:
            st.session_state.error_message = error_msg
            st.session_state.analysis_result = None
            st.session_state.is_loading = False
        else:
            st.session_state.error_message = None
            st.session_state.is_loading = True
            try:
                start_time = time.time()
                prediction = run_analysis_with_loader(article_text)
                analysis_duration = time.time() - start_time
                log_prediction(article_text, prediction, analysis_duration)
                st.session_state.analysis_result = prediction
            except Exception as e:
                st.session_state.error_message = f"Analysis failed: {str(e)}"
                st.session_state.analysis_result = None
            finally:
                st.session_state.is_loading = False

    if st.session_state.error_message:
        render_error(st.session_state.error_message)

    if st.session_state.analysis_result:
        render_results(st.session_state.analysis_result)
        footer_html = """
        <div style="text-align: center; background-color: #dbeafe; font-size: 0.6rem; border-radius: 10px; padding: 2px;">
            <h5 style="margin: 0.5rem 0; font-size: 0.8rem; opacity: 0.9; padding-bottom: 6px;">
            <span style= "font-weight: bold">Important:</span> This tool uses machine learning to estimate credibility and is not infallible. Always cross-check news with primary sources, established fact-checkers, and official reporting.
            </h5>
        </div>
        """
        st.markdown(footer_html, unsafe_allow_html=True)
    render_info_section()


if __name__ == "__main__":
    main()
    