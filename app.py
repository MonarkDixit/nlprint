#app.py
#NLPrint: Amazon Review Authenticity Detector
#Author: Monark Dixit (UID: 122259645)
#Course: MSML606, Spring 2026

#PURPOSE:
#This is the Streamlit frontend for NLPrint. It connects the full backend pipeline (preprocessing, MinHash, index, query engine) to an interactive web UI that allows any user to:

#1. Paste or type any Amazon review text
#2. Submit it for similarity analysis
#3. See the top-5 most similar reviews from the indexed dataset with similarity percentage badges
#4. See metadata flags surfacing suspicious signals:
#   - Shared user_id (same account posted multiple similar reviews)
#   - Timestamp proximity (reviews posted very close together)
#   - Verified purchase mismatches
#5. See algorithm stats in the sidebar:
#    - Hash table load factor
#    - MinHash parameters (k, shingle size)
#    - Index size and build time
#    - Query latency

import streamlit as st
import time
import os
import sys

#Add project root to path so src imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.indexer import SignatureIndex, INDEX_SAVE_PATH
from src.query import QueryEngine
from src.minhash import DEFAULT_K
from src.preprocessor import SHINGLE_SIZE
from src.hasher import DEFAULT_TABLE_SIZE


#PAGE CONFIGURATION
#Must be the first Streamlit call in the script.

st.set_page_config(
    page_title="NLPrint | Review Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


#CUSTOM CSS
#Injects styling to give the app a clean, dark forensic-tool aesthetic. Uses a monospace-forward design that feels analytical and trustworthy.

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a2235;
        --bg-card-hover: #1f2a42;
        --accent-primary: #3b82f6;
        --accent-secondary: #06b6d4;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
        --accent-success: #10b981;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #475569;
        --border-color: #1e293b;
        --border-accent: #2d3f5e;
    }

    /* Global background */
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main content area */
    .main .block-container {
        padding: 2rem 2.5rem 4rem 2.5rem;
        max-width: 1200px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: var(--text-primary) !important;
    }
    p, li, label {
        color: var(--text-secondary);
        font-family: 'DM Sans', sans-serif;
    }

    /* Header banner */
    .nlprint-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid var(--border-accent);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .nlprint-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    }
    .nlprint-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
    }
    .nlprint-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: var(--text-secondary);
        margin: 0;
        font-weight: 300;
    }
    .nlprint-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: var(--accent-primary);
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin-top: 0.8rem;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-accent) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }
    .stTextArea label {
        color: var(--text-secondary) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Submit button */
    .stButton button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 2rem !important;
        letter-spacing: 0.05em !important;
        transition: opacity 0.2s !important;
        width: 100%;
        text-shadow: 0 1px 2px rgba(0,0,0,0.4) !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .stButton button:hover {
        opacity: 0.9 !important;
    }

    /* Result cards */
    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--border-accent);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        transition: border-color 0.2s;
    }
    .result-card:hover {
        border-color: var(--accent-primary);
    }
    .result-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }
    .result-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: var(--text-primary);
        font-weight: 700;
        flex: 1;
        margin-right: 1rem;
    }
    .similarity-badge {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .sim-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #ef4444;
    }
    .sim-medium {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #f59e0b;
    }
    .sim-low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10b981;
    }
    .result-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }
    .result-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
    }
    .meta-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border-color);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }

    /* Flag chips */
    .flag-chip {
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin: 0.2rem 0.2rem 0.2rem 0;
        font-weight: 700;
    }
    .flag-danger {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }
    .flag-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #f59e0b;
    }
    .flag-info {
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid rgba(6, 182, 212, 0.3);
        color: #06b6d4;
    }

    /* Section labels */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border-color);
    }

    /* Stats sidebar */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border-color);
    }
    .stat-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        color: var(--text-muted);
    }
    .stat-value {
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
        color: var(--accent-secondary);
        font-weight: 700;
    }

    /* Info box */
    .info-box {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .info-box p {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.6;
    }

    /* No results state */
    .no-results {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-muted);
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
    }

    /* Star rating display */
    .stars {
        color: #f59e0b;
        font-size: 0.85rem;
    }

    /* Query info strip */
    .query-info-strip {
        background: var(--bg-card);
        border: 1px solid var(--border-accent);
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .query-info-item {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    .query-info-value {
        color: var(--accent-secondary);
        font-weight: 700;
    }

    /* Flags panel */
    .flags-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-accent);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.5rem;
    }

    /* Divider */
    hr {
        border-color: var(--border-color) !important;
    }

    /* Spinner */
    .stSpinner {
        color: var(--accent-primary) !important;
    }

    /* Selectbox and other inputs */
    .stSelectbox select {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-accent) !important;
    }
</style>
""", unsafe_allow_html=True)


#INDEX LOADING
#Cached so it only runs once per session. On first run, builds and saves the index. On subsequent runs, loads it instantly from disk.

@st.cache_resource(show_spinner=False)
def load_index():
    """
    Loads or builds the signature index. Cached across Streamlit reruns.
    Returns a tuple of (SignatureIndex, QueryEngine).
    """
    index = SignatureIndex(
        k=DEFAULT_K,
        shingle_size=SHINGLE_SIZE,
        table_size=DEFAULT_TABLE_SIZE,
        seed=42,
    )
    index.load_or_build(verbose=False)
    engine = QueryEngine(index)
    return index, engine


#HELPER FUNCTIONS
def get_similarity_class(similarity: float) -> str:
    #Returns the CSS class for the similarity badge based on the score.
    if similarity >= 0.5:
        return "sim-high"
    elif similarity >= 0.2:
        return "sim-medium"
    else:
        return "sim-low"


def get_flag_class(flag: str) -> str:
    #Returns the CSS class for a metadata flag chip.
    if flag == "shared_user_id":
        return "flag-danger"
    elif flag in ("timestamp_proximity", "verified_purchase_mismatch"):
        return "flag-warning"
    else:
        return "flag-info"


def render_stars(rating) -> str:
    #Converts a numeric rating to a star string for display.
    if rating is None:
        return ""
    try:
        r = int(float(rating))
        return "★" * r + "☆" * (5 - r)
    except (ValueError, TypeError):
        return ""


def render_flag_chip(flag: str) -> str:
    #Returns an HTML flag chip for a given flag string.
    css_class = get_flag_class(flag)
    labels = {
        "shared_user_id": "SHARED ACCOUNT",
        "timestamp_proximity": "CLOSE TIMESTAMP",
        "verified_purchase_mismatch": "PURCHASE MISMATCH",
    }
    label = labels.get(flag, flag.upper().replace("_", " "))
    return f'<span class="flag-chip {css_class}">{label}</span>'


def render_result_card(result: dict, rank: int) -> str:
    #Renders a single result card as an HTML string.
    sim_class = get_similarity_class(result["similarity"])
    stars_html = render_stars(result["rating"])

    #Truncate long review text for display
    display_text = result["text"]
    if len(display_text) > 300:
        display_text = display_text[:300] + "..."

    title_display = result["title"] if result["title"] else "No title"

    #Build flag chips HTML
    flags_html = ""
    for flag in result["metadata_flags"]:
        flags_html += render_flag_chip(flag)

    #Build metadata chips
    vp = "Verified" if result["verified_purchase"] else "Not Verified"
    meta_html = f"""
        <span class="meta-chip">Rating: {stars_html} {result['rating'] or 'N/A'}</span>
        <span class="meta-chip">{vp}</span>
        <span class="meta-chip">Date: {result['timestamp_display']}</span>
        <span class="meta-chip">Helpful: {result['helpful_vote']}</span>
        <span class="meta-chip">Shingles: {result['shingle_count']}</span>
    """

    #User ID (partially masked for display)
    uid = result["user_id"]
    uid_display = uid[:8] + "..." if len(uid) > 8 else uid

    card_html = f"""
    <div class="result-card">
        <div class="result-card-header">
            <div>
                <div style="font-family: 'Space Mono', monospace; font-size: 0.7rem;
                            color: #475569; margin-bottom: 0.3rem;">
                    #{rank} MATCH &nbsp;|&nbsp; USER: {uid_display}
                </div>
                <div class="result-title">{title_display}</div>
            </div>
            <div class="similarity-badge {sim_class}">{result['similarity_pct']}</div>
        </div>
        <div class="result-text">{display_text}</div>
        <div class="result-meta">{meta_html}</div>
        {f'<div style="margin-top: 0.8rem;">{flags_html}</div>' if flags_html else ''}
    </div>
    """
    return card_html


#SIDEBAR
def render_sidebar(index: SignatureIndex, query_response: dict = None):
    #Renders the sidebar with algorithm parameters, index stats, and last query info.
    with st.sidebar:
        st.markdown("""
        <div style="font-family: 'Space Mono', monospace; font-size: 1rem;
                    font-weight: 700; color: #f1f5f9; margin-bottom: 0.3rem;">
            NLPrint
        </div>
        <div style="font-family: 'DM Sans', sans-serif; font-size: 0.8rem;
                    color: #475569; margin-bottom: 1.5rem;">
            Review Authenticity Detector
        </div>
        """, unsafe_allow_html=True)

        stats = index.get_stats()

        st.markdown('<div class="section-label">ALGORITHM PARAMETERS</div>',
                    unsafe_allow_html=True)

        param_rows = [
            ("MinHash functions (k)", str(stats["k"])),
            ("Shingle size", f"{stats['shingle_size']} chars"),
            ("Hash table size", f"{stats['table_size']} buckets"),
            ("Avg load factor", f"{stats['avg_load_factor']:.3f}"),
            ("Avg shingle count", f"{stats['avg_shingle_count']:.0f}"),
        ]
        for label, value in param_rows:
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-label">{label}</span>
                <span class="stat-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<br><div class="section-label">INDEX STATS</div>',
                    unsafe_allow_html=True)

        index_rows = [
            ("Reviews indexed", f"{stats['entry_count']:,}"),
            ("Build time", f"{stats['build_time_seconds']:.1f}s"),
        ]
        for label, value in index_rows:
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-label">{label}</span>
                <span class="stat-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        if query_response and query_response.get("query_time_ms"):
            st.markdown('<br><div class="section-label">LAST QUERY</div>',
                        unsafe_allow_html=True)
            query_rows = [
                ("Query latency", f"{query_response['query_time_ms']:.1f}ms"),
                ("Shingles in input", str(query_response["shingle_count"])),
                ("Language detected", query_response["language"]),
                ("Results found", str(len(query_response["results"]))),
            ]
            for label, value in query_rows:
                st.markdown(f"""
                <div class="stat-row">
                    <span class="stat-label">{label}</span>
                    <span class="stat-value">{value}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">HOW IT WORKS</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'DM Sans', sans-serif; font-size: 0.8rem;
                    color: #475569; line-height: 1.7;">
            NLPrint converts each review into overlapping character trigrams
            (shingles), then applies MinHash to produce a compact signature.
            Similarity is estimated by comparing signatures without ever
            doing brute-force pairwise comparison.
            <br><br>
            Complexity: O(n&middot;k) vs O(n&sup2;)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family: 'Space Mono', monospace; font-size: 0.65rem;
                    color: #334155; text-align: center;">
            MSML606 Spring 2026<br>
            Monark Dixit | UID: 122259645
        </div>
        """, unsafe_allow_html=True)


#MAIN APP
def main():
    #Load index (cached)
    with st.spinner("Loading NLPrint index..."):
        index, engine = load_index()

    #Header
    st.markdown("""
    <div class="nlprint-header">
        <div class="nlprint-title">🔍 NLPrint</div>
        <div class="nlprint-subtitle">
            Amazon Review Authenticity Detector &nbsp;&nbsp;|&nbsp;&nbsp;
            MinHash + Feature Hashing &nbsp;&nbsp;|&nbsp;&nbsp;
            All_Beauty Dataset (10,000+ Reviews)
        </div>
        <div class="nlprint-badge">MSML606 &nbsp;|&nbsp; Spring 2026 &nbsp;|&nbsp; Monark Dixit</div>
    </div>
    """, unsafe_allow_html=True)

    #Initialize session state
    if "query_response" not in st.session_state:
        st.session_state.query_response = None
    if "submitted_text" not in st.session_state:
        st.session_state.submitted_text = ""

    #Sidebar (uses last query response for latency stats)
    render_sidebar(index, st.session_state.query_response)

    #Main layout: two columns
    left_col, right_col = st.columns([1, 1.6], gap="large")

    #LEFT COLUMN: Input panel
    with left_col:
        st.markdown('<div class="section-label">REVIEW INPUT</div>',
                    unsafe_allow_html=True)

        review_text = st.text_area(
            label="Paste or type any Amazon review below",
            placeholder=(
                "Example: This moisturizer is absolutely amazing. "
                "My skin has never felt so soft and hydrated. "
                "I have been using it every morning and evening for two weeks "
                "and the results are incredible. Highly recommend to everyone!"
            ),
            height=220,
            key="review_input",
        )

        top_n = st.select_slider(
            "Number of results to show",
            options=[3, 5, 7, 10],
            value=5,
        )

        submit_clicked = st.button("ANALYZE REVIEW", use_container_width=True)

        #How to use box
        st.markdown("""
        <div class="info-box">
            <p>
                Paste any Amazon beauty product review and NLPrint will find
                the most similar reviews in its database using MinHash
                approximate similarity search. High similarity scores
                (above 50%) may indicate copy-pasted or fraudulent content.
            </p>
        </div>
        """, unsafe_allow_html=True)

        #Example reviews section
        st.markdown('<div class="section-label" style="margin-top:1rem;">TRY AN EXAMPLE</div>',
                    unsafe_allow_html=True)

        example_reviews = {
            "Positive review": (
                "This product is absolutely amazing. I have been using it for "
                "two weeks and my skin looks incredible. The texture is smooth "
                "and it absorbs quickly. Highly recommend to everyone!"
            ),
            "Negative review": (
                "Terrible product. It broke out my skin and smells awful. "
                "Waste of money. Would not recommend this to anyone."
            ),
            "Short review": (
                "Smells good, feels great! Love it."
            ),
        }

        for label, example_text in example_reviews.items():
            if st.button(label, key=f"example_{label}", use_container_width=True):
                st.session_state.submitted_text = example_text
                response = engine.query(example_text, top_n=top_n)
                st.session_state.query_response = response
                st.rerun()

    #RIGHT COLUMN: Results panel
    with right_col:

        #Handle submit button
        if submit_clicked and review_text.strip():
            with st.spinner("Analyzing review..."):
                response = engine.query(review_text.strip(), top_n=top_n)
                st.session_state.query_response = response
                st.session_state.submitted_text = review_text.strip()

        response = st.session_state.query_response

        #Results panel
        if response is None:
            #Empty state
            st.markdown("""
            <div class="no-results">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🔍</div>
                <div>Enter a review on the left and click ANALYZE REVIEW</div>
                <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #334155;">
                    Results will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif response.get("error"):
            st.error(response["error"])

        else:
            results = response["results"]

            # Query info strip
            flags_str = ", ".join(response["flags"]) if response["flags"] else "none"
            st.markdown(f"""
            <div class="query-info-strip">
                <div class="query-info-item">
                    SHINGLES: <span class="query-info-value">{response['shingle_count']}</span>
                </div>
                <div class="query-info-item">
                    LANGUAGE: <span class="query-info-value">{response['language'].upper()}</span>
                </div>
                <div class="query-info-item">
                    LATENCY: <span class="query-info-value">{response['query_time_ms']:.1f}ms</span>
                </div>
                <div class="query-info-item">
                    SEARCHED: <span class="query-info-value">{response['index_size']:,} reviews</span>
                </div>
                <div class="query-info-item">
                    FLAGS: <span class="query-info-value">{flags_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not results:
                st.markdown("""
                <div class="no-results">
                    <div style="font-size: 1.5rem; margin-bottom: 0.8rem;">🟢</div>
                    <div>No similar reviews found above the threshold.</div>
                    <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #334155;">
                        This review appears to be unique in the dataset.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="section-label">TOP {len(results)} SIMILAR REVIEWS</div>',
                    unsafe_allow_html=True
                )

                # Render each result card
                for rank, result in enumerate(results, 1):
                    card_html = render_result_card(result, rank)
                    st.markdown(card_html, unsafe_allow_html=True)

                # Metadata flags summary panel
                flagged_results = [r for r in results if r["has_flags"]]
                if flagged_results:
                    st.markdown("""
                    <div class="flags-panel">
                        <div class="section-label">SUSPICIOUS SIGNAL SUMMARY</div>
                    """, unsafe_allow_html=True)

                    for result in flagged_results:
                        flags_html = "".join(
                            render_flag_chip(f) for f in result["metadata_flags"]
                        )
                        uid = result["user_id"]
                        uid_display = uid[:12] + "..." if len(uid) > 12 else uid
                        st.markdown(f"""
                        <div style="margin-bottom: 0.8rem; padding-bottom: 0.8rem;
                                    border-bottom: 1px solid #1e293b;">
                            <div style="font-family: 'Space Mono', monospace;
                                        font-size: 0.72rem; color: #475569;
                                        margin-bottom: 0.4rem;">
                                {uid_display} &nbsp;|&nbsp;
                                {result['timestamp_display']} &nbsp;|&nbsp;
                                {result['similarity_pct']} match
                            </div>
                            {flags_html}
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)


#ENTRY POINT

if __name__ == "__main__":
    main()