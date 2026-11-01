"""
===================================================================
ANLP5 – NLTK WordNet Explorer | Full Web Application
Advanced Natural Language Processing (AML23702) CIA Lab – Question 5
===================================================================
"""

import os
import streamlit as st
import pandas as pd

# Import utilities
from utils.nltk_setup import ensure_nltk_resources
from utils.helpers import (
    POS_MAP, POS_COLORS, POS_BADGES, POS_ICONS,
    clean_word_input, is_valid_word_query, get_pos_label
)

# Import backend modules
from modules.wordnet_explorer import get_word_overview, get_synsets, get_synset_details
from modules.synonym_extractor import extract_synonyms_by_meaning, extract_all_unique_synonyms
from modules.definition_extractor import extract_definitions
from modules.word_sense import get_bank_case_study, get_showcase_comparison, POLYSEMOUS_SHOWCASES
from modules.context_analyzer import analyze_context_meaning
from modules.visualizations import (
    create_pos_distribution_chart,
    create_lemmas_per_sense_chart,
    create_taxonomy_depth_chart,
    create_sunburst_hierarchy_chart,
    create_lesk_overlap_chart
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Injection
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ANLP5 – WordNet Explorer",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
css_file_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_file_path):
    with open(css_file_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Ensure NLTK resources
nltk_status = ensure_nltk_resources()

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "searched_word" not in st.session_state:
    st.session_state.searched_word = "bank"

if "context_word" not in st.session_state:
    st.session_state.context_word = "bank"

if "context_sentence" not in st.session_state:
    st.session_state.context_sentence = "The river bank was muddy after the heavy rainfall."

if "context_pos" not in st.session_state:
    st.session_state.context_pos = "All Parts of Speech"

if "nav_module" not in st.session_state:
    st.session_state.nav_module = "🏠 Dashboard / Overview"

# Helper to change active word
def set_active_word(word: str, nav_to: str = None):
    st.session_state.searched_word = word
    st.session_state.context_word = word
    if nav_to:
        st.session_state.nav_module = nav_to


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Global Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding-bottom: 0.8rem; border-bottom: 1px solid #E2E8F0; margin-bottom: 1rem;">
            <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #6366F1; letter-spacing: 0.08em;">
                AML23702 – CIA Question 5
            </div>
            <div style="font-size: 1.35rem; font-weight: 800; color: #0F172A; font-family: 'Outfit', sans-serif;">
                ANLP5 WordNet Explorer
            </div>
            <div style="font-size: 0.82rem; color: #64748B;">
                Synonym & Word Sense Disambiguation
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Radio
    nav_options = [
        "🏠 Dashboard / Overview",
        "🔍 Word Search",
        "📚 Synonym Explorer",
        "📖 Definition Explorer",
        "⚖️ Word Sense Disambiguation",
        "🎯 Contextual Meaning Analysis",
        "📊 WordNet Visualizations",
        "💡 Sample Demonstrations"
    ]
    
    current_index = nav_options.index(st.session_state.nav_module) if st.session_state.nav_module in nav_options else 0
    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=current_index,
        label_visibility="collapsed"
    )
    st.session_state.nav_module = selected_nav

    st.markdown("---")
    
    # Global Quick Search
    st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#334155; margin-bottom:0.3rem;'>⚡ Quick Word Search</p>", unsafe_allow_html=True)
    with st.form(key="sidebar_search_form", clear_on_submit=False):
        sidebar_query = st.text_input("Enter word", value=st.session_state.searched_word, label_visibility="collapsed", placeholder="e.g. bank, light, apple...")
        submit_search = st.form_submit_button("Search WordNet", use_container_width=True)
        if submit_search:
            valid, msg = is_valid_word_query(sidebar_query)
            if valid:
                set_active_word(clean_word_input(sidebar_query))
                st.rerun()
            else:
                st.error(msg)

    # Sample Quick Links
    st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#334155; margin-top:1.2rem; margin-bottom:0.4rem;'>💡 Benchmark Words</p>", unsafe_allow_html=True)
    sample_cols = st.columns(2)
    sample_words = ["bank", "book", "apple", "light", "bat", "run", "computer"]
    for i, w in enumerate(sample_words):
        col = sample_cols[i % 2]
        if col.button(w, key=f"sb_sample_{w}", use_container_width=True):
            set_active_word(w)
            st.rerun()

    # System Status Footer in Sidebar
    st.markdown("---")
    if nltk_status["success"]:
        st.markdown("""
            <div style="font-size: 0.78rem; color: #10B981; display: flex; align-items: center; gap: 0.4rem;">
                <span>🟢</span> <b>NLTK WordNet 3.0 Active</b>
            </div>
            <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 0.2rem;">
                Corpus verified & loaded locally
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ WordNet data loading issue. Check internet or local corpus.")


# Fetch data for current word
current_word = st.session_state.searched_word
overview = get_word_overview(current_word)


# =============================================================================
# MODULE 1: Dashboard / Overview
# =============================================================================
if st.session_state.nav_module == "🏠 Dashboard / Overview":
    # Hero Section
    st.markdown(f"""
        <div class="academic-hero">
            <div class="hero-tag">🎓 Advanced NLP Laboratory • Experiment 5</div>
            <div class="hero-title">Synonym & Word Sense Identification using NLTK WordNet</div>
            <div class="hero-subtitle">
                An interactive computational linguistics dashboard to explore word meanings, lexical relations,
                synonyms, definitions, and transparent word sense disambiguation using the Princeton WordNet database.
            </div>
            <div class="hero-meta">
                <span>📘 <strong>Course:</strong> Advanced NLP (AML23702)</span>
                <span>🔬 <strong>Lexical Engine:</strong> NLTK WordNet</span>
                <span>⚙️ <strong>Current Word:</strong> <code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px; color: #FFF;">{current_word}</code></span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Core Academic Information Cards
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
            <div class="synset-card">
                <h3 style="margin-top:0; color:#1E1B4B; font-size:1.25rem;">📌 Problem Statement & Objective</h3>
                <p style="color:#334155; font-size:0.95rem; line-height:1.6;">
                    <strong>Problem Statement:</strong> Write a Python program using NLTK WordNet to identify the synonyms and
                    definitions of a given word and distinguish its possible meanings in different contexts.
                </p>
                <p style="color:#334155; font-size:0.95rem; line-height:1.6;">
                    <strong>Objective:</strong> Develop an interactive, genuine web application that allows users to explore:
                </p>
                <ul style="color:#475569; font-size:0.92rem; line-height:1.6; margin-bottom:0;">
                    <li><strong>Synset Discovery:</strong> Cognitive synonym sets across distinct parts of speech.</li>
                    <li><strong>Sense Glosses:</strong> Precise definitions and corpus example sentences.</li>
                    <li><strong>Lexical Richness:</strong> Lemmatized synonyms grouped by semantic concept.</li>
                    <li><strong>Word Sense Disambiguation (WSD):</strong> Distinguishing polysemous meanings (e.g., <em>Financial Bank</em> vs. <em>River Bank</em>) using transparent context-gloss overlap.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="synset-card">
                <h3 style="margin-top:0; color:#1E1B4B; font-size:1.25rem;">🛠️ Technology Stack & Architecture</h3>
                <div style="display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding-bottom: 0.4rem;">
                        <span style="font-weight: 600; color: #334155;">Python 3.13</span>
                        <span style="color: #64748B;">Core Programming Language</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding-bottom: 0.4rem;">
                        <span style="font-weight: 600; color: #334155;">NLTK (Natural Language Toolkit)</span>
                        <span style="color: #64748B;">Corpus Reader & Stemmers</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding-bottom: 0.4rem;">
                        <span style="font-weight: 600; color: #334155;">Princeton WordNet 3.0</span>
                        <span style="color: #64748B;">Lexical Semantic Database</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding-bottom: 0.4rem;">
                        <span style="font-weight: 600; color: #334155;">Streamlit</span>
                        <span style="color: #64748B;">Interactive Web Application Framework</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #F1F5F9; padding-bottom: 0.4rem;">
                        <span style="font-weight: 600; color: #334155;">Plotly</span>
                        <span style="color: #64748B;">Interactive NLP Data Visualizations</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600; color: #334155;">Custom CSS</span>
                        <span style="color: #64748B;">Modern Academic Design System</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # WordNet Educational Concept Guide
    st.markdown("<h3 style='margin-top:1.5rem; margin-bottom:1rem; color:#0F172A;'>📚 Fundamentals of Princeton WordNet</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">📦</div>
                <div style="font-size:1.05rem; font-weight:700; color:#1E293B; margin-bottom:0.3rem;">What is a Synset?</div>
                <div style="font-size:0.85rem; color:#64748B; line-height:1.5;">
                    A <strong>Synset</strong> (synonym set) is a set of cognitive synonyms representing a single distinct concept or meaning.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">🏷️</div>
                <div style="font-size:1.05rem; font-weight:700; color:#1E293B; margin-bottom:0.3rem;">What is a Lemma?</div>
                <div style="font-size:0.85rem; color:#64748B; line-height:1.5;">
                    A <strong>Lemma</strong> is the canonical dictionary headword or specific morphological spelling attached to a synset.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">🌳</div>
                <div style="font-size:1.05rem; font-weight:700; color:#1E293B; margin-bottom:0.3rem;">Semantic Hierarchy</div>
                <div style="font-size:0.85rem; color:#64748B; line-height:1.5;">
                    Nouns and verbs are organized into hierarchies connected by <strong>Hypernyms</strong> (super-ordinate) and <strong>Hyponyms</strong> (sub-ordinate).
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">⚖️</div>
                <div style="font-size:1.05rem; font-weight:700; color:#1E293B; margin-bottom:0.3rem;">Word Senses</div>
                <div style="font-size:0.85rem; color:#64748B; line-height:1.5;">
                    <strong>Polysemy</strong> occurs when a single word has multiple related or unrelated senses depending on its syntactic and semantic context.
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Instructions for use
    st.markdown("""
        <div class="academic-callout info" style="margin-top:2rem;">
            <strong>🚀 Quick Start Instructions:</strong>
            <ol style="margin-top:0.4rem; margin-bottom:0; padding-left:1.2rem;">
                <li>Use the <strong>Quick Word Search</strong> in the sidebar or navigate to <strong>Word Search</strong> to input any English term.</li>
                <li>Explore <strong>Synonym Explorer</strong> to see synonyms grouped strictly by their semantic sense (not an unstructured list).</li>
                <li>Examine <strong>Definition Explorer</strong> for structured glosses, lemmas, and taxonomy depth.</li>
                <li>Review <strong>Word Sense Disambiguation</strong> for the classic <em>Bank</em> case study and polysemy demonstrations.</li>
                <li>Test <strong>Contextual Meaning Analysis</strong> with your own custom sentences using the explainable Simplified Lesk algorithm.</li>
                <li>Inspect <strong>WordNet Visualizations</strong> for interactive distribution charts and sunburst hierarchies.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    # Action buttons
    act_col1, act_col2 = st.columns([1, 1])
    with act_col1:
        if st.button("🔍 Search WordNet for Active Word ('" + current_word + "')", use_container_width=True, type="primary"):
            st.session_state.nav_module = "🔍 Word Search"
            st.rerun()
    with act_col2:
        if st.button("⚖️ Jump to Bank Case Study (WSD)", use_container_width=True):
            set_active_word("bank", "⚖️ Word Sense Disambiguation")
            st.rerun()


# =============================================================================
# MODULE 2: Word Search
# =============================================================================
elif st.session_state.nav_module == "🔍 Word Search":
    st.markdown("""
        <div class="module-header">
            <div class="module-title">🔍 Word Search & Lexical Overview</div>
            <div class="module-desc">Enter any English word to retrieve real-time synsets, senses, and lexical metrics from NLTK WordNet.</div>
        </div>
    """, unsafe_allow_html=True)

    # Search Bar
    with st.container():
        s_col1, s_col2 = st.columns([4, 1])
        with s_col1:
            search_input = st.text_input(
                "Search English Word",
                value=current_word,
                placeholder="Type an English word (e.g., bank, light, book, bat, computer)...",
                label_visibility="collapsed"
            )
        with s_col2:
            search_btn = st.button("Search", use_container_width=True, type="primary")

        if search_btn:
            valid, msg = is_valid_word_query(search_input)
            if valid:
                set_active_word(clean_word_input(search_input))
                st.rerun()
            else:
                st.error(msg)

    # WordNet Availability and Result
    if not overview["found"]:
        st.markdown(f"""
            <div class="academic-callout warning">
                <strong>Word Not Found:</strong> No synset entries were found for <code>{current_word}</code> in the WordNet database.
                <br><em>Please verify your spelling or test another benchmark word such as <strong>bank, apple, book, run, light</strong>.</em>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Availability Banner
        st.markdown(f"""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:0.8rem 1.2rem; margin:1rem 0; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="color:#15803D; font-weight:700; font-size:1.05rem;">WordNet Entry Confirmed:</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:#14532D; font-weight:700; margin-left:0.4rem;">{overview['word']}</span>
                </div>
                <div style="font-size:0.82rem; color:#166534; background:#DCFCE7; padding:0.25rem 0.65rem; border-radius:9999px; font-weight:600;">
                    ✓ {overview['total_synsets']} Synset Senses Identified
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 4 Hero Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📑</div>
                    <div class="metric-value">{overview['total_synsets']}</div>
                    <div class="metric-label">Synsets (Senses)</div>
                    <div class="metric-sub">Distinct cognitive concepts</div>
                </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">✨</div>
                    <div class="metric-value">{overview['unique_synonyms_count']}</div>
                    <div class="metric-label">Unique Synonyms</div>
                    <div class="metric-sub">Across all WordNet senses</div>
                </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🏷️</div>
                    <div class="metric-value">{len(overview['pos_counts'])}</div>
                    <div class="metric-label">Parts of Speech</div>
                    <div class="metric-sub">{', '.join([get_pos_label(p) for p in overview['pos_counts'].keys()])}</div>
                </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">💬</div>
                    <div class="metric-value">{overview['total_examples']}</div>
                    <div class="metric-label">Example Sentences</div>
                    <div class="metric-sub">Curated in WordNet glosses</div>
                </div>
            """, unsafe_allow_html=True)

        # Primary Sense Highlight Card
        primary = overview["primary_sense"]
        if primary:
            st.markdown(f"""
                <div class="synset-card" style="margin-top:1.5rem; border-left: 4px solid #4F46E5;">
                    <div class="synset-card-header">
                        <span style="font-weight:700; color:#4338CA; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.06em;">
                            ⭐ Most Frequent Sense (Primary WordNet Sense)
                        </span>
                        <div>
                            <span class="synset-id-badge">{primary['name']}</span>
                            <span class="pos-badge {POS_BADGES.get(primary['pos'], 'badge-noun')}">{primary['pos_label']}</span>
                        </div>
                    </div>
                    <div class="synset-definition">"{primary['definition']}"</div>
                    {f'<div class="synset-examples"><span class="example-label">Example</span>"{primary["examples"][0]}"</div>' if primary["examples"] else ''}
                    <div>
                        <span style="font-size:0.82rem; font-weight:600; color:#475569;">Lemmas in this synset: </span>
                        <span style="font-size:0.84rem; color:#334155;">{', '.join(primary['lemmas'])}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Structured Table of All Senses
        st.markdown("<h4 style='margin-top:1.5rem; margin-bottom:0.8rem; color:#1E293B;'>📋 Complete Synset Sense Roster</h4>", unsafe_allow_html=True)
        table_rows = []
        for i, d in enumerate(overview["synset_details"], start=1):
            table_rows.append({
                "Sense #": i,
                "Synset ID": d["name"],
                "Part of Speech": d["pos_label"],
                "Definition": d["definition"],
                "Synonyms (Lemmas)": ", ".join(d["lemmas"]),
                "Taxonomy Depth": f"{d['min_depth']} - {d['max_depth']}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# =============================================================================
# MODULE 3: Synonym Explorer
# =============================================================================
elif st.session_state.nav_module == "📚 Synonym Explorer":
    st.markdown(f"""
        <div class="module-header">
            <div class="module-title">📚 Synonym Explorer for '{current_word}'</div>
            <div class="module-desc">Retrieve real synonyms grouped strictly by their semantic meaning rather than an unstructured list.</div>
        </div>
    """, unsafe_allow_html=True)

    if not overview["found"]:
        st.warning(f"No WordNet entries found for '{current_word}'. Please search for a valid word.")
    else:
        # Part of Speech Filter Controls
        pos_options = ["All Parts of Speech"] + [f"{get_pos_label(k)} ({v})" for k, v in overview["pos_counts"].items()]
        selected_pos_label = st.selectbox("Filter Senses by Part of Speech:", options=pos_options)

        pos_code_filter = None
        if selected_pos_label != "All Parts of Speech":
            # Extract pos code from selected label
            for k in overview["pos_counts"].keys():
                if get_pos_label(k) in selected_pos_label:
                    pos_code_filter = k
                    break

        meaning_groups = extract_synonyms_by_meaning(current_word, pos_filter=pos_code_filter)

        # Overview of Unique Synonyms Cloud
        unique_synonyms = overview["unique_synonyms"]
        st.markdown(f"""
            <div class="synset-card" style="background:#F8FAFC; border: 1px dashed #CBD5E1; margin-bottom:1.5rem;">
                <div style="font-weight:700; color:#1E293B; margin-bottom:0.4rem; font-size:0.95rem;">
                    🌐 All Unique Synonyms Across Senses ({len(unique_synonyms)} found):
                </div>
                <div class="synonym-chips-container">
                    {''.join([f'<span class="synonym-chip">{s}</span>' for s in unique_synonyms]) if unique_synonyms else '<span style="color:#94A3B8; font-size:0.85rem;">No other distinct synonyms found in WordNet for this word.</span>'}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<h4 style='color:#0F172A; margin-bottom:1rem;'>📑 Synonyms Grouped by Sense ({len(meaning_groups)} Senses)</h4>", unsafe_allow_html=True)

        # Loop through each meaning group and render an individual card
        for idx, group in enumerate(meaning_groups, start=1):
            badge_class = POS_BADGES.get(group["pos"], "badge-noun")
            
            # Format synonym chips
            chips_html = []
            for syn in group["synonyms"]:
                is_target = (syn.lower() == current_word.lower())
                chip_class = "synonym-chip target-word" if is_target else "synonym-chip"
                chips_html.append(f'<span class="{chip_class}">{"⭐ " if is_target else ""}{syn}</span>')

            chips_rendered = "".join(chips_html)

            # Examples HTML
            examples_html = ""
            if group["examples"]:
                ex_items = "".join([f'<li>"{ex}"</li>' for ex in group["examples"]])
                examples_html = f"""
                    <div class="synset-examples">
                        <span class="example-label">Corpus Examples</span>
                        <ul style="margin:0; padding-left:1.2rem; font-style:italic;">{ex_items}</ul>
                    </div>
                """

            st.markdown(f"""
                <div class="synset-card">
                    <div class="synset-card-header">
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span style="font-weight:700; color:#64748B; font-size:0.9rem;">#{idx}</span>
                            <span class="synset-id-badge">{group['synset_name']}</span>
                            <span class="pos-badge {badge_class}">{group['pos_label']}</span>
                        </div>
                        <div style="font-size:0.8rem; color:#94A3B8;">
                            {len(group['synonyms'])} Synonyms in this synset
                        </div>
                    </div>
                    <div class="synset-definition">
                        <strong>Meaning:</strong> {group['definition']}
                    </div>
                    {examples_html}
                    <div>
                        <div style="font-size:0.82rem; font-weight:700; color:#475569; margin-bottom:0.3rem; text-transform:uppercase; letter-spacing:0.04em;">
                            Associated Synonyms (Lemmas):
                        </div>
                        <div class="synonym-chips-container">
                            {chips_rendered}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MODULE 4: Definition Explorer
# =============================================================================
elif st.session_state.nav_module == "📖 Definition Explorer":
    st.markdown(f"""
        <div class="module-header">
            <div class="module-title">📖 Definition Explorer for '{current_word}'</div>
            <div class="module-desc">Detailed inspection of WordNet definitions, glosses, example sentences, lemmas, and semantic hierarchy.</div>
        </div>
    """, unsafe_allow_html=True)

    if not overview["found"]:
        st.warning(f"No WordNet entries found for '{current_word}'.")
    else:
        # Search & Filters for definitions
        def_col1, def_col2 = st.columns([2, 1])
        with def_col1:
            keyword_search = st.text_input("Filter definitions by keyword:", placeholder="e.g. water, institution, money, flight, aircraft...")
        with def_col2:
            pos_filter_options = ["All"] + [get_pos_label(k) for k in overview["pos_counts"].keys()]
            def_pos_choice = st.selectbox("Filter by POS:", options=pos_filter_options)

        pos_code = None
        if def_pos_choice != "All":
            for k in overview["pos_counts"].keys():
                if get_pos_label(k) == def_pos_choice:
                    pos_code = k
                    break

        definitions_list = extract_definitions(
            current_word,
            pos_filter=pos_code,
            search_term=keyword_search if keyword_search.strip() else None
        )

        st.markdown(f"<p style='color:#64748B; font-size:0.9rem;'>Displaying <strong>{len(definitions_list)}</strong> of {overview['total_synsets']} total definitions.</p>", unsafe_allow_html=True)

        for d in definitions_list:
            badge_class = POS_BADGES.get(d["pos"], "badge-noun")
            
            # Lemmas chips
            lemmas_html = "".join([f'<span class="synonym-chip">{lm}</span>' for lm in d["lemmas"]])
            
            # Examples
            ex_html = ""
            if d["examples"]:
                ex_list = "".join([f"<li>\"{e}\"</li>" for e in d["examples"]])
                ex_html = f"""
                    <div class="synset-examples">
                        <span class="example-label">Example Sentences in WordNet</span>
                        <ul style="margin:0; padding-left:1.2rem;">{ex_list}</ul>
                    </div>
                """
            else:
                ex_html = """
                    <div style="font-size:0.82rem; color:#94A3B8; font-style:italic; margin-bottom:0.8rem;">
                        No example sentence available in WordNet gloss for this sense.
                    </div>
                """

            # Hypernyms pills
            hypernyms_html = ""
            if d["hypernyms"]:
                h_pills = "".join([f'<span class="relation-pill">↑ {h}</span>' for h in d["hypernyms"][:5]])
                hypernyms_html = f"""
                    <div style="margin-top:0.6rem;">
                        <span style="font-size:0.8rem; font-weight:600; color:#64748B;">Direct Hypernyms (Super-concepts):</span>
                        <div style="margin-top:0.2rem;">{h_pills}</div>
                    </div>
                """

            st.markdown(f"""
                <div class="synset-card">
                    <div class="synset-card-header">
                        <div style="display:flex; align-items:center; gap:0.5rem;">
                            <span class="synset-id-badge">ID: {d['synset_id']}</span>
                            <span class="pos-badge {badge_class}">{d['pos_label']}</span>
                        </div>
                        <div style="font-size:0.8rem; color:#64748B;">
                            Taxonomy Depth: <strong>{d['min_depth']} - {d['max_depth']}</strong> | Hyponyms: <strong>{d['hyponym_count']}</strong>
                        </div>
                    </div>
                    <div class="synset-definition">
                        "{d['definition']}"
                    </div>
                    {ex_html}
                    <div>
                        <span style="font-size:0.8rem; font-weight:600; color:#64748B;">Lemmas in this Synset:</span>
                        <div class="synonym-chips-container">
                            {lemmas_html}
                        </div>
                    </div>
                    {hypernyms_html}
                </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MODULE 5: Word Sense Disambiguation
# =============================================================================
elif st.session_state.nav_module == "⚖️ Word Sense Disambiguation":
    st.markdown("""
        <div class="module-header">
            <div class="module-title">⚖️ Word Sense Disambiguation (WSD)</div>
            <div class="module-desc">Demonstrating lexical polysemy and distinguishing possible meanings in different contexts using NLTK WordNet.</div>
        </div>
    """, unsafe_allow_html=True)

    # Architectural Callout
    st.markdown("""
        <div class="academic-callout info">
            <strong>⚠️ Core Computational Linguistics Principle:</strong>
            <br>
            <strong>Princeton WordNet</strong> provides a static, curated <em>Sense Inventory</em> (the list of distinct meanings, synsets, and definitions for words).
            WordNet alone does <strong>not</strong> perform automatic context disambiguation. Contextual disambiguation requires an algorithmic layer
            (such as the Lesk algorithm or representation embeddings) to determine which specific WordNet synset applies to a word in a given sentence.
        </div>
    """, unsafe_allow_html=True)

    # Primary Benchmark: "bank"
    st.markdown("<h3 style='color:#0F172A; margin-top:1.5rem;'>🏛️ Foundational Case Study: The Senses of 'bank'</h3>", unsafe_allow_html=True)
    st.markdown("The English noun **'bank'** is the quintessential NLP demonstration for polysemy and homonymy:")

    bank_study = get_bank_case_study()
    p_contrast = bank_study["primary_contrast"]

    # Side by side contrast cards
    bank_c1, bank_c2 = st.columns(2)

    with bank_c1:
        s1 = p_contrast["sense_1"]
        d1 = s1["details"]
        st.markdown(f"""
            <div class="wsd-card" style="border-top: 4px solid #059669; height:100%;">
                <div style="font-size:0.78rem; font-weight:700; color:#059669; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">
                    Sense 1 • Geographical
                </div>
                <div style="font-size:1.3rem; font-weight:800; color:#0F172A; font-family:'Outfit',sans-serif; margin-bottom:0.4rem;">
                    {s1['title']}
                </div>
                <div style="margin-bottom:0.75rem;">
                    <span class="synset-id-badge">{d1['name']}</span>
                    <span class="pos-badge badge-noun">Noun</span>
                </div>
                <div style="font-size:0.95rem; color:#1E293B; margin-bottom:0.8rem; line-height:1.5;">
                    <strong>WordNet Definition:</strong> "{d1['definition']}"
                </div>
                <div style="background:#F0FDF4; border-left:3px solid #059669; padding:0.6rem 0.8rem; border-radius:0 6px 6px 0; margin-bottom:0.8rem;">
                    <span style="font-size:0.78rem; font-weight:700; color:#047857; text-transform:uppercase; display:block;">Context Sentence:</span>
                    <span style="font-size:0.88rem; color:#064E3B; font-style:italic;">"{s1['context_example']}"</span>
                </div>
                <div>
                    <span style="font-size:0.8rem; font-weight:600; color:#64748B;">Key Context Clues:</span>
                    <div style="margin-top:0.3rem;">
                        {''.join([f'<span class="overlap-token-badge">{w}</span>' for w in s1['key_discriminators']])}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with bank_c2:
        s2 = p_contrast["sense_2"]
        d2 = s2["details"]
        st.markdown(f"""
            <div class="wsd-card" style="border-top: 4px solid #2563EB; height:100%;">
                <div style="font-size:0.78rem; font-weight:700; color:#2563EB; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;">
                    Sense 2 • Financial
                </div>
                <div style="font-size:1.3rem; font-weight:800; color:#0F172A; font-family:'Outfit',sans-serif; margin-bottom:0.4rem;">
                    {s2['title']}
                </div>
                <div style="margin-bottom:0.75rem;">
                    <span class="synset-id-badge">{d2['name']}</span>
                    <span class="pos-badge badge-noun">Noun</span>
                </div>
                <div style="font-size:0.95rem; color:#1E293B; margin-bottom:0.8rem; line-height:1.5;">
                    <strong>WordNet Definition:</strong> "{d2['definition']}"
                </div>
                <div style="background:#EFF6FF; border-left:3px solid #2563EB; padding:0.6rem 0.8rem; border-radius:0 6px 6px 0; margin-bottom:0.8rem;">
                    <span style="font-size:0.78rem; font-weight:700; color:#1D4ED8; text-transform:uppercase; display:block;">Context Sentence:</span>
                    <span style="font-size:0.88rem; color:#1E3A8A; font-style:italic;">"{s2['context_example']}"</span>
                </div>
                <div>
                    <span style="font-size:0.8rem; font-weight:600; color:#64748B;">Key Context Clues:</span>
                    <div style="margin-top:0.3rem;">
                        {''.join([f'<span class="overlap-token-badge">{w}</span>' for w in s2['key_discriminators']])}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Additional Bank Senses
    st.markdown("<h4 style='color:#1E293B; margin-top:1.5rem;'>✈️ Other Specialized Senses of 'bank' in WordNet</h4>", unsafe_allow_html=True)
    extra_cols = st.columns(len(bank_study["additional_senses"]))
    for i, extra in enumerate(bank_study["additional_senses"]):
        with extra_cols[i]:
            ed = extra["details"]
            st.markdown(f"""
                <div class="synset-card" style="height:100%;">
                    <span class="synset-id-badge" style="margin-bottom:0.5rem;">{ed['name']}</span>
                    <div style="font-weight:700; color:#1E293B; font-size:1rem; margin-bottom:0.3rem;">{extra['title']}</div>
                    <div style="font-size:0.85rem; color:#475569; margin-bottom:0.6rem;">"{ed['definition']}"</div>
                    <div style="font-size:0.82rem; color:#0369A1; font-style:italic;">"{extra['context_example']}"</div>
                </div>
            """, unsafe_allow_html=True)

    # Additional Pre-configured Polysemous Pairs
    st.markdown("<h3 style='color:#0F172A; margin-top:2.5rem;'>🔍 Interactive Polysemy Showcases</h3>", unsafe_allow_html=True)
    poly_word_selected = st.selectbox(
        "Select an ambiguous word to inspect its dual meanings:",
        options=list(POLYSEMOUS_SHOWCASES.keys()),
        format_func=lambda x: f"{x.capitalize()} – {POLYSEMOUS_SHOWCASES[x]['description']}"
    )

    showcase_data = get_showcase_comparison(poly_word_selected)
    if showcase_data:
        p_c1, p_c2 = st.columns(2)
        sa = showcase_data["sense_a"]
        sb = showcase_data["sense_b"]

        with p_c1:
            st.markdown(f"""
                <div class="synset-card" style="border-top:3px solid #6366F1; height:100%;">
                    <div class="synset-id-badge" style="margin-bottom:0.4rem;">{sa['details']['name']}</div>
                    <div style="font-size:1.15rem; font-weight:700; color:#1E293B; margin-bottom:0.4rem;">{sa['title']}</div>
                    <div style="font-size:0.9rem; color:#334155; margin-bottom:0.8rem;">"{sa['details']['definition']}"</div>
                    <div class="synset-examples">
                        <span class="example-label">Example Sentence:</span>
                        "{sa['sentence']}"
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with p_c2:
            st.markdown(f"""
                <div class="synset-card" style="border-top:3px solid #EC4899; height:100%;">
                    <div class="synset-id-badge" style="margin-bottom:0.4rem;">{sb['details']['name']}</div>
                    <div style="font-size:1.15rem; font-weight:700; color:#1E293B; margin-bottom:0.4rem;">{sb['title']}</div>
                    <div style="font-size:0.9rem; color:#334155; margin-bottom:0.8rem;">"{sb['details']['definition']}"</div>
                    <div class="synset-examples">
                        <span class="example-label">Example Sentence:</span>
                        "{sb['sentence']}"
                    </div>
                </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MODULE 6: Contextual Meaning Analysis
# =============================================================================
elif st.session_state.nav_module == "🎯 Contextual Meaning Analysis":
    st.markdown("""
        <div class="module-header">
            <div class="module-title">🎯 Contextual Meaning Analysis (Simplified Lesk Algorithm)</div>
            <div class="module-desc">Transparent, explainable disambiguation comparing user sentence context with WordNet synset glosses.</div>
        </div>
    """, unsafe_allow_html=True)

    # Explanation of the method
    st.markdown("""
        <div class="academic-callout info">
            <strong>Transparent Methodology (Simplified Lesk Algorithm):</strong>
            <ol style="margin-top:0.3rem; margin-bottom:0; padding-left:1.2rem;">
                <li><strong>Tokenization & Filtering:</strong> Normalizes the context sentence, removes stopwords and punctuation, and isolates surrounding content words.</li>
                <li><strong>Morphological Lemmatization:</strong> Expands inflections (e.g. <em>deposited</em> → <em>deposit</em>) to align with dictionary base forms.</li>
                <li><strong>Gloss Intersection:</strong> For every candidate WordNet synset, extracts the gloss (definition + examples + synonyms) and computes the exact set intersection of content words.</li>
                <li><strong>Explainable Ranking:</strong> Ranks synsets by the count of shared tokens and Jaccard similarity. If no tokens match (overlap = 0), transparently reports the Most Frequent Sense (MFS) fallback.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    # Input Form
    with st.container():
        st.markdown("<h4 style='margin-bottom:0.5rem; color:#0F172A;'>1. Input Target Ambiguous Word & Context Sentence</h4>", unsafe_allow_html=True)
        
        ctx_word_col, ctx_pos_col = st.columns([3, 1])
        with ctx_word_col:
            target_input = st.text_input("Target Ambiguous Word:", value=st.session_state.context_word, placeholder="e.g. bank, light, book, bat...")
        pos_options_list = ["All Parts of Speech", "Noun (n)", "Verb (v)", "Adjective (a)", "Adverb (r)"]
        default_pos_idx = pos_options_list.index(st.session_state.context_pos) if st.session_state.context_pos in pos_options_list else 0
        with ctx_pos_col:
            pos_choice = st.selectbox("POS Filter (Optional):", options=pos_options_list, index=default_pos_idx)
            st.session_state.context_pos = pos_choice

        # Quick Preset Sentences
        st.markdown("<p style='font-size:0.82rem; font-weight:600; color:#64748B; margin-bottom:0.3rem;'>Preset Context Examples:</p>", unsafe_allow_html=True)
        preset_cols = st.columns(3)
        if preset_cols[0].button("🌊 River Bank: 'The river bank was muddy after the heavy rainfall.'"):
            st.session_state.context_word = "bank"
            st.session_state.context_sentence = "The river bank was muddy after the heavy rainfall."
            st.session_state.context_pos = "Noun (n)"
            st.rerun()
        if preset_cols[1].button("💰 Financial Bank: 'I deposited money into my bank account yesterday.'"):
            st.session_state.context_word = "bank"
            st.session_state.context_sentence = "I deposited money into my bank account yesterday."
            st.session_state.context_pos = "Noun (n)"
            st.rerun()
        if preset_cols[2].button("🦇 Nocturnal Bat: 'The nocturnal bat flew out of the dark cave.'"):
            st.session_state.context_word = "bat"
            st.session_state.context_sentence = "The nocturnal bat flew out of the dark cave."
            st.session_state.context_pos = "Noun (n)"
            st.rerun()

        sentence_input = st.text_area(
            "Context Sentence containing the target word:",
            value=st.session_state.context_sentence,
            rows=2,
            placeholder="Type a full sentence using the target word..."
        )

        analyze_btn = st.button("🚀 Analyze Context Meaning", type="primary", use_container_width=True)

    if analyze_btn or sentence_input:
        pos_code_opt = None
        if "Noun" in pos_choice:
            pos_code_opt = "n"
        elif "Verb" in pos_choice:
            pos_code_opt = "v"
        elif "Adjective" in pos_choice:
            pos_code_opt = "a"
        elif "Adverb" in pos_choice:
            pos_code_opt = "r"

        analysis = analyze_context_meaning(
            target_word=target_input,
            context_sentence=sentence_input,
            pos_filter=pos_code_opt
        )

        if not analysis["success"]:
            st.error(analysis["error"])
        else:
            # Tokenization Inspector
            st.markdown("<h4 style='margin-top:1.5rem; color:#1E293B;'>2. Feature Extraction & Content Token Breakdown</h4>", unsafe_allow_html=True)
            tok_col1, tok_col2 = st.columns([1, 1])
            with tok_col1:
                st.markdown(f"""
                    <div class="synset-card">
                        <div style="font-weight:700; color:#334155; font-size:0.9rem; margin-bottom:0.4rem;">Extracted Context Content Tokens:</div>
                        <div>{''.join([f'<span class="synonym-chip">{t}</span>' for t in analysis['context_tokens']]) if analysis['context_tokens'] else '<em>None</em>'}</div>
                        <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.4rem;">(Stopwords, punctuation, and target word '{analysis['target_word']}' excluded)</div>
                    </div>
                """, unsafe_allow_html=True)

            with tok_col2:
                st.markdown(f"""
                    <div class="synset-card">
                        <div style="font-weight:700; color:#334155; font-size:0.9rem; margin-bottom:0.4rem;">Lemmatized Token Vocabulary:</div>
                        <div>{''.join([f'<span class="synonym-chip" style="background:#EFF6FF; border-color:#BFDBFE;">{t}</span>' for t in analysis['context_lemmatized']])}</div>
                        <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.4rem;">(Normalized base forms for flexible dictionary matching)</div>
                    </div>
                """, unsafe_allow_html=True)

            # Winner / Best Match Card
            best = analysis["best_match"]
            st.markdown("<h4 style='margin-top:1.5rem; color:#1E293B;'>3. Disambiguation Result</h4>", unsafe_allow_html=True)

            if analysis["is_confident"]:
                st.markdown(f"""
                    <div class="wsd-card winner">
                        <div class="winner-banner">✓ Best Contextual Match Identified</div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; flex-wrap:wrap;">
                            <span class="synset-id-badge" style="font-size:1.1rem; padding:0.35rem 0.8rem;">{best['synset_id']}</span>
                            <span class="pos-badge {POS_BADGES.get(best['pos'], 'badge-noun')}">{best['pos_label']}</span>
                        </div>
                        <div style="font-size:1.15rem; font-weight:600; color:#0F172A; margin-bottom:0.8rem; line-height:1.5;">
                            "{best['definition']}"
                        </div>
                        {f'<div class="synset-examples"><span class="example-label">Synset Example</span>"{best["examples"][0]}"</div>' if best["examples"] else ''}
                        <div style="background:#FFFFFF; border:1px solid #D1FAE5; border-radius:10px; padding:0.8rem 1rem; margin-top:0.8rem;">
                            <div style="font-size:0.85rem; font-weight:700; color:#065F46; margin-bottom:0.4rem;">
                                🎯 Overlapping Clue Words ({best['overlap_count']} matched):
                            </div>
                            <div>
                                {''.join([f'<span class="overlap-token-badge">{w}</span>' for w in best['overlap_tokens']])}
                            </div>
                            <div style="font-size:0.8rem; color:#047857; margin-top:0.4rem;">
                                Jaccard Similarity Score: <strong>{best['jaccard_score']}</strong>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="wsd-card" style="border:2px solid #F59E0B; background:#FFFBEB;">
                        <div style="font-weight:700; color:#B45309; font-size:0.95rem; margin-bottom:0.4rem;">
                            ⚠️ Zero Context-Gloss Overlap Detected
                        </div>
                        <div style="font-size:0.9rem; color:#92400E; line-height:1.5; margin-bottom:0.6rem;">
                            None of the content words in your sentence matched the WordNet definitions or examples for <strong>'{analysis['target_word']}'</strong>.
                        </div>
                        <div style="font-size:0.85rem; color:#78350F; background:#FEF3C7; padding:0.6rem 0.8rem; border-radius:8px;">
                            <strong>NLP Fallback Rule:</strong> When no contextual evidence is available, the standard NLP heuristic defaults to WordNet's
                            <strong>Most Frequent Sense (MFS)</strong>: <code>{best['synset_id']}</code> ("{best['definition']}").
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Interactive Plotly Chart of Candidates
            st.plotly_chart(create_lesk_overlap_chart(analysis), use_container_width=True)

            # Complete Breakdown Table
            st.markdown("<h4 style='margin-top:1.5rem; color:#1E293B;'>4. Candidate Synset Evaluation Table</h4>", unsafe_allow_html=True)
            cand_table = []
            for c in analysis["candidate_results"]:
                cand_table.append({
                    "Rank": c["rank"],
                    "Synset ID": c["synset_id"],
                    "POS": c["pos_label"],
                    "Overlap Count": c["overlap_count"],
                    "Matched Words": ", ".join(c["overlap_tokens"]) if c["overlap_tokens"] else "-",
                    "Jaccard Score": c["jaccard_score"],
                    "Definition": c["definition"]
                })
            st.dataframe(pd.DataFrame(cand_table), use_container_width=True, hide_index=True)


# =============================================================================
# MODULE 7: WordNet Visualizations
# =============================================================================
elif st.session_state.nav_module == "📊 WordNet Visualizations":
    st.markdown(f"""
        <div class="module-header">
            <div class="module-title">📊 WordNet Interactive Visualizations for '{current_word}'</div>
            <div class="module-desc">Dynamic Plotly charts constructed entirely from retrieved NLTK WordNet data.</div>
        </div>
    """, unsafe_allow_html=True)

    if not overview["found"]:
        st.warning(f"No WordNet entries available for '{current_word}'.")
    else:
        # Chart Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🍩 Part of Speech Distribution",
            "📊 Synonyms per Sense (Lexical Richness)",
            "🌳 Taxonomy Hierarchy Depth",
            "☀️ Lexical Hierarchy Sunburst"
        ])

        with tab1:
            st.plotly_chart(create_pos_distribution_chart(overview), use_container_width=True)
            st.markdown("""
                <div style="font-size:0.85rem; color:#64748B; padding:0 0.5rem;">
                    <strong>About this chart:</strong> Illustrates the distribution of the word's synsets across syntactic categories
                    (Nouns, Verbs, Adjectives, Adverbs). Words with high cross-categorical presence exhibit extensive syntactic flexibility.
                </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.plotly_chart(create_lemmas_per_sense_chart(overview), use_container_width=True)
            st.markdown("""
                <div style="font-size:0.85rem; color:#64748B; padding:0 0.5rem;">
                    <strong>About this chart:</strong> Measures the number of unique lemmas (synonyms) grouped under each synset.
                    Senses with higher lemma counts represent broader semantic clusters with rich lexical alternatives.
                </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.plotly_chart(create_taxonomy_depth_chart(overview), use_container_width=True)
            st.markdown("""
                <div style="font-size:0.85rem; color:#64748B; padding:0 0.5rem;">
                    <strong>About this chart:</strong> Measures the shortest and longest paths from the root entity (<code>entity.n.01</code>)
                    to the synset in the WordNet is-a hierarchy. Deeper synsets denote specialized, fine-grained concepts, while shallower synsets denote general categories.
                </div>
            """, unsafe_allow_html=True)

        with tab4:
            st.plotly_chart(create_sunburst_hierarchy_chart(overview), use_container_width=True)
            st.markdown("""
                <div style="font-size:0.85rem; color:#64748B; padding:0 0.5rem;">
                    <strong>About this chart:</strong> Multi-tier drilldown visualizing the relation:
                    <code>Word → Part of Speech → Synset Sense → Individual Lemmas</code>. Click on any segment to zoom in or zoom out.
                </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MODULE 8: Sample Demonstrations
# =============================================================================
elif st.session_state.nav_module == "💡 Sample Demonstrations":
    st.markdown("""
        <div class="module-header">
            <div class="module-title">💡 Curated Benchmark Demonstrations</div>
            <div class="module-desc">Ready-to-use sample words exhibiting classic NLP phenomena (polysemy, homonymy, cross-POS flexibility).</div>
        </div>
    """, unsafe_allow_html=True)

    benchmark_samples = [
        {
            "word": "bank",
            "title": "Classic Polysemy & Homonymy",
            "tag": "18 Synsets • Noun & Verb",
            "desc": "The primary NLP case study contrasting financial monetary institutions with geographical river slopes, flight banking maneuvers, and reserve funds.",
            "sample_senses": ["River slope (bank.n.01)", "Financial institution (depository_financial_institution.n.01)", "Aircraft maneuver (bank.n.10)"]
        },
        {
            "word": "book",
            "title": "Physical Object vs Action",
            "tag": "15 Synsets • Noun & Verb",
            "desc": "Exhibits polysemy between bound printed publication, literary composition, ledger record, and the verb to reserve tickets or accommodations.",
            "sample_senses": ["Physical publication (book.n.01)", "Written work (book.n.02)", "Make reservation (book.v.01)"]
        },
        {
            "word": "apple",
            "title": "Fruit vs Cultivated Tree",
            "tag": "2 Synsets • Noun",
            "desc": "A clean semantic distinction between the edible orchard pome fruit and the native Eurasian malus apple tree.",
            "sample_senses": ["Edible fruit (apple.n.01)", "Apple tree (apple.n.02)"]
        },
        {
            "word": "light",
            "title": "Cross-POS Polysemy",
            "tag": "61 Synsets • Noun, Verb & Adjective",
            "desc": "Extreme syntactic diversity spanning visible electromagnetic radiation, ignition sources, low physical weight, and mental illumination.",
            "sample_senses": ["Visible radiation (light.n.01)", "Low weight (light.a.01)", "Ignite / kindle (light.v.01)"]
        },
        {
            "word": "bat",
            "title": "Zoological vs Sports Club",
            "tag": "11 Synsets • Noun & Verb",
            "desc": "Contrasts the nocturnal winged mammal (Chiroptera) with the wooden sports equipment used in baseball and cricket.",
            "sample_senses": ["Nocturnal mammal (bat.n.01)", "Sports club (bat.n.05)", "Strike with bat (bat.v.01)"]
        },
        {
            "word": "run",
            "title": "Massive Verbal & Nominal Polysemy",
            "tag": "57 Synsets • Noun & Verb",
            "desc": "One of the most polysemous verbs in the English language, covering fast locomotion, operating software, political campaigning, and baseball scores.",
            "sample_senses": ["Fast sprint (run.v.01)", "Operate machine (run.v.19)", "Baseball score (run.n.05)"]
        },
        {
            "word": "computer",
            "title": "Modern Machine vs Human Role",
            "tag": "2 Synsets • Noun",
            "desc": "Captures the historical transition from human computational experts (calculators) to electronic programmable computing machines.",
            "sample_senses": ["Electronic digital machine (computer.n.01)", "Human calculator (calculator.n.02)"]
        }
    ]

    for item in benchmark_samples:
        b_col1, b_col2 = st.columns([3, 1])
        with b_col1:
            st.markdown(f"""
                <div class="synset-card" style="margin-bottom:1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                        <span style="font-size:1.3rem; font-weight:800; color:#0F172A; font-family:'Outfit',sans-serif;">
                            {item['word'].capitalize()}
                        </span>
                        <span class="synset-id-badge">{item['tag']}</span>
                    </div>
                    <div style="font-weight:600; color:#4F46E5; font-size:0.92rem; margin-bottom:0.4rem;">
                        {item['title']}
                    </div>
                    <p style="font-size:0.88rem; color:#475569; line-height:1.5; margin-bottom:0.6rem;">
                        {item['desc']}
                    </p>
                    <div style="font-size:0.8rem; color:#64748B;">
                        <strong>Key Senses in WordNet:</strong> {', '.join(item['sample_senses'])}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with b_col2:
            st.write("")
            st.write("")
            if st.button(f"🚀 Load '{item['word']}'", key=f"btn_demo_{item['word']}", use_container_width=True, type="primary"):
                set_active_word(item["word"], "🔍 Word Search")
                st.rerun()


# -----------------------------------------------------------------------------
# 4. Global Footer
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="academic-footer">
        <strong>ANLP5 – NLTK WordNet Explorer</strong> • Advanced Natural Language Processing (AML23702) CIA Question 5<br>
        Developed with Python, Streamlit, NLTK WordNet, Pandas, & Plotly. Real-time lexical database retrieval.
    </div>
""", unsafe_allow_html=True)
