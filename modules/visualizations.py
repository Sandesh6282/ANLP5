"""
Visualizations Module for ANLP5 WordNet Explorer.
Generates interactive, publication-grade Plotly visualizations
based strictly on real WordNet retrieved data.
Includes POS distribution, lexical richness per sense, taxonomy depth,
sunburst hierarchy, and contextual overlap rankings.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import POS_MAP, POS_COLORS, get_pos_label


CHART_THEME = {
    "font_family": "Inter, -apple-system, sans-serif",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "title_font": {"size": 16, "family": "Outfit, Inter, sans-serif", "color": "#0F172A"},
    "margin": dict(l=40, r=30, t=50, b=40)
}


def create_pos_distribution_chart(overview: Dict[str, Any]) -> go.Figure:
    """
    Creates an interactive donut chart showing the breakdown of synsets across Parts of Speech.
    """
    pos_counts = overview.get("pos_counts", {})
    if not pos_counts:
        fig = go.Figure()
        fig.add_annotation(text="No WordNet data available", showarrow=False, font=dict(size=14, color="#64748B"))
        return fig

    labels = [get_pos_label(k) for k in pos_counts.keys()]
    values = list(pos_counts.values())
    colors = [POS_COLORS.get(k, '#64748B') for k in pos_counts.keys()]

    total_synsets = sum(values)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Proportion: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title=dict(text=f"<b>Part of Speech Distribution</b> ({overview['word']})", font=CHART_THEME["title_font"]),
        annotations=[dict(
            text=f"<b>{total_synsets}</b><br><span style='font-size:11px;color:#64748B;'>Synsets</span>",
            x=0.5, y=0.5,
            font_size=20,
            font_family="Outfit, sans-serif",
            showarrow=False
        )],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=CHART_THEME["margin"],
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=dict(family=CHART_THEME["font_family"])
    )

    return fig


def create_lemmas_per_sense_chart(overview: Dict[str, Any]) -> go.Figure:
    """
    Horizontal bar chart displaying the number of synonyms (lemmas) for each WordNet sense.
    Illustrates lexical richness across different senses.
    """
    details = overview.get("synset_details", [])
    if not details:
        fig = go.Figure()
        fig.add_annotation(text="No synsets available", showarrow=False)
        return fig

    # Limit to top 15 synsets if word has many senses (like 'run') for readability
    capped_details = details[:15]

    synset_names = [d["name"] for d in capped_details][::-1]
    lemma_counts = [len(d["lemmas"]) for d in capped_details][::-1]
    pos_codes = [d["pos"] for d in capped_details][::-1]
    definitions = [d["definition"] for d in capped_details][::-1]
    lemmas_list = [", ".join(d["lemmas"][:5]) + ("..." if len(d["lemmas"]) > 5 else "") for d in capped_details][::-1]
    bar_colors = [POS_COLORS.get(p, '#6366F1') for p in pos_codes]

    fig = go.Figure(go.Bar(
        x=lemma_counts,
        y=synset_names,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color='#E2E8F0', width=1)),
        customdata=list(zip(definitions, lemmas_list)),
        hovertemplate="<b>%{y}</b><br>Synonyms Count: %{x}<br><b>Lemmas:</b> %{customdata[1]}<br><i>%{customdata[0]}</i><extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Synonyms per Word Sense (Lexical Richness)</b>", font=CHART_THEME["title_font"]),
        xaxis_title="Number of Unique Lemmas (Synonyms)",
        yaxis_title="WordNet Synset",
        margin=dict(l=160, r=30, t=50, b=40),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor="#F8FAFC",
        font=dict(family=CHART_THEME["font_family"]),
        xaxis=dict(gridcolor="#E2E8F0", showgrid=True, dtick=1)
    )

    return fig


def create_taxonomy_depth_chart(overview: Dict[str, Any]) -> go.Figure:
    """
    Bar chart showing taxonomy hierarchy depth (path distance to root 'entity.n.01').
    Differentiates between abstract (shallow depth) and fine-grained (deep depth) senses.
    """
    details = overview.get("synset_details", [])
    if not details:
        fig = go.Figure()
        fig.add_annotation(text="No taxonomy data available", showarrow=False)
        return fig

    capped_details = details[:12]
    synset_names = [d["name"] for d in capped_details]
    max_depths = [d["max_depth"] for d in capped_details]
    min_depths = [d["min_depth"] for d in capped_details]
    pos_labels = [d["pos_label"] for d in capped_details]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Min Depth to Root',
        x=synset_names,
        y=min_depths,
        marker_color='#93C5FD',
        hovertemplate="<b>%{x}</b><br>Min Depth: %{y}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        name='Max Depth to Root',
        x=synset_names,
        y=max_depths,
        marker_color='#2563EB',
        hovertemplate="<b>%{x}</b><br>Max Depth: %{y}<extra></extra>"
    ))

    fig.update_layout(
        barmode='group',
        title=dict(text="<b>WordNet Taxonomy Hierarchy Depth</b> (Specificity Level)", font=CHART_THEME["title_font"]),
        xaxis_title="Synset",
        yaxis_title="Depth in WordNet Taxonomy Tree",
        xaxis_tickangle=-35,
        margin=dict(l=40, r=30, t=50, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor="#F8FAFC",
        font=dict(family=CHART_THEME["font_family"]),
        yaxis=dict(gridcolor="#E2E8F0", showgrid=True)
    )

    return fig


def create_sunburst_hierarchy_chart(overview: Dict[str, Any]) -> go.Figure:
    """
    Interactive Sunburst chart displaying lexical hierarchy:
    Searched Word -> Part of Speech -> Synset -> Lemmas (Synonyms).
    """
    word = overview.get("word", "")
    details = overview.get("synset_details", [])
    
    if not details:
        fig = go.Figure()
        fig.add_annotation(text="No hierarchy available", showarrow=False)
        return fig

    # Build hierarchical tree ids, labels, and parents
    ids = [word]
    labels = [word.capitalize()]
    parents = [""]
    values = [0]

    # Level 1: POS categories
    pos_set = set(d["pos"] for d in details)
    for pos_code in sorted(list(pos_set)):
        pos_id = f"{word}_{pos_code}"
        pos_label = get_pos_label(pos_code)
        ids.append(pos_id)
        labels.append(pos_label)
        parents.append(word)
        values.append(0)

    # Level 2 & 3: Synsets and their Lemmas (limit to first 10 synsets for visual clarity)
    for d in details[:10]:
        synset_id = d["name"]
        pos_id = f"{word}_{d['pos']}"
        ids.append(synset_id)
        labels.append(d["name"].split('.')[0] + f" ({d['name'].split('.')[-1]})")
        parents.append(pos_id)
        values.append(len(d["lemmas"]))

        # Level 3: Lemmas
        for lm in d["lemmas"][:4]:  # Top 4 lemmas per synset
            lemma_id = f"{synset_id}_{lm}"
            ids.append(lemma_id)
            labels.append(lm)
            parents.append(synset_id)
            values.append(1)

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        maxdepth=3,
        hovertemplate="<b>%{label}</b><br>Branch weight: %{value}<extra></extra>",
        marker=dict(colorscale="Blues")
    ))

    fig.update_layout(
        title=dict(text=f"<b>WordNet Lexical Hierarchy</b> (Click to Zoom)", font=CHART_THEME["title_font"]),
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=dict(family=CHART_THEME["font_family"])
    )

    return fig


def create_lesk_overlap_chart(context_results: Dict[str, Any]) -> go.Figure:
    """
    Bar chart showing candidate synset ranking and overlap token scores
    from the Contextual Meaning Analysis (Lesk Algorithm).
    """
    candidates = context_results.get("candidate_results", [])
    if not candidates:
        fig = go.Figure()
        fig.add_annotation(text="No candidates evaluated", showarrow=False)
        return fig

    # Limit to top 10 candidates
    top_cands = candidates[:10]
    synsets = [c["synset_id"] for c in top_cands][::-1]
    overlaps = [c["overlap_count"] for c in top_cands][::-1]
    jaccards = [c["jaccard_score"] for c in top_cands][::-1]
    matched_words = [", ".join(c["overlap_tokens"]) if c["overlap_tokens"] else "None" for c in top_cands][::-1]

    # Color winner in emerald green, others in indigo
    best_id = context_results.get("best_match", {}).get("synset_id", "")
    colors = ['#10B981' if s == best_id and o > 0 else '#6366F1' for s, o in zip(synsets, overlaps)]

    fig = go.Figure(go.Bar(
        x=overlaps,
        y=synsets,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#CBD5E1', width=1)),
        customdata=list(zip(matched_words, jaccards)),
        hovertemplate="<b>%{y}</b><br>Overlap Tokens: %{x}<br><b>Matched Words:</b> %{customdata[0]}<br>Jaccard Score: %{customdata[1]}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Contextual Candidate Overlap Ranking (Lesk Algorithm)</b>", font=CHART_THEME["title_font"]),
        xaxis_title="Number of Matching Context Content Words",
        yaxis_title="Candidate Synset",
        margin=dict(l=180, r=30, t=50, b=40),
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor="#F8FAFC",
        font=dict(family=CHART_THEME["font_family"]),
        xaxis=dict(gridcolor="#E2E8F0", showgrid=True, dtick=1)
    )

    return fig
