"""
Contextual Meaning Analysis Module for ANLP5.
Implements a transparent, explainable Simplified Lesk Algorithm for Word Sense Disambiguation (WSD).
Compares context sentences with WordNet synset definitions, examples, and lemmas.
Provides step-by-step token breakdown, matching intersections, and similarity metrics.
"""

from typing import List, Dict, Any, Optional, Set
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from modules.wordnet_explorer import get_synsets, get_synset_details
from utils.helpers import clean_word_input, tokenize_and_filter, get_stopwords


def _get_lemmatizer():
    """Lazily load or reuse WordNetLemmatizer."""
    try:
        return WordNetLemmatizer()
    except Exception:
        return None


def get_lemmatized_tokens(tokens: List[str]) -> Set[str]:
    """
    Lemmatizes tokens across noun and verb categories to ensure inflected words
    (e.g., 'deposits', 'deposited', 'flying', 'flew') match base dictionary entries.
    """
    lemmatizer = _get_lemmatizer()
    token_set = set(tokens)
    if not lemmatizer:
        return token_set

    expanded = set(token_set)
    for token in token_set:
        try:
            expanded.add(lemmatizer.lemmatize(token, pos='n'))
            expanded.add(lemmatizer.lemmatize(token, pos='v'))
            expanded.add(lemmatizer.lemmatize(token, pos='a'))
        except Exception:
            pass
    return expanded


def analyze_context_meaning(
    target_word: str,
    context_sentence: str,
    pos_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs transparent, explainable Simplified Lesk Word Sense Disambiguation.

    Parameters:
    - target_word: The ambiguous word to disambiguate (e.g., 'bank')
    - context_sentence: The user-supplied sentence containing the target word
    - pos_filter: Optional POS filter ('n', 'v', 'a', 'r')

    Returns:
    - structured dictionary with:
      - target_word
      - context_sentence
      - context_tokens (extracted content words excluding the target word)
      - total_candidates
      - candidate_results (sorted by overlap score descending)
      - best_match (highest overlap synset)
      - is_confident (True if overlap > 0)
      - methodology_notes
    """
    cleaned_word = clean_word_input(target_word)
    stops = get_stopwords()
    
    # 1. Tokenize and clean context sentence
    raw_context_tokens = tokenize_and_filter(context_sentence)
    
    # Exclude the target word itself from context tokens to prevent tautological matches
    context_content_tokens = [
        tok for tok in raw_context_tokens
        if tok.lower() != cleaned_word.lower() and tok.lower() not in stops
    ]
    
    context_lemmatized_set = get_lemmatized_tokens(context_content_tokens)

    # 2. Retrieve all candidate synsets
    candidate_synsets = get_synsets(cleaned_word, pos=pos_filter)
    if not candidate_synsets:
        return {
            "success": False,
            "error": f"No WordNet synsets found for '{cleaned_word}'.",
            "target_word": cleaned_word,
            "context_sentence": context_sentence,
            "context_tokens": context_content_tokens,
            "total_candidates": 0,
            "candidate_results": [],
            "best_match": None,
            "is_confident": False
        }

    candidate_results = []

    for synset in candidate_synsets:
        details = get_synset_details(synset)
        
        # Build gloss string: definition + examples + other lemma synonyms
        gloss_parts = [synset.definition()]
        gloss_parts.extend(synset.examples())
        # Include lemma names (excluding target word itself)
        other_lemmas = [
            l.replace('_', ' ') for l in details["lemmas"]
            if l.lower() != cleaned_word.lower()
        ]
        gloss_parts.extend(other_lemmas)
        
        full_gloss_text = " ".join(gloss_parts)
        
        # Tokenize and lemmatize gloss
        raw_gloss_tokens = tokenize_and_filter(full_gloss_text)
        gloss_content_tokens = [
            tok for tok in raw_gloss_tokens
            if tok.lower() != cleaned_word.lower() and tok.lower() not in stops
        ]
        gloss_lemmatized_set = get_lemmatized_tokens(gloss_content_tokens)
        
        # Exact token overlap (intersection)
        overlap_tokens = sorted(list(context_lemmatized_set.intersection(gloss_lemmatized_set)))
        overlap_count = len(overlap_tokens)
        
        # Calculate Jaccard similarity between context tokens and gloss tokens
        union_count = len(context_lemmatized_set.union(gloss_lemmatized_set))
        jaccard_score = round(overlap_count / union_count, 4) if union_count > 0 else 0.0

        candidate_results.append({
            "synset_id": synset.name(),
            "pos": details["pos"],
            "pos_label": details["pos_label"],
            "definition": details["definition"],
            "examples": details["examples"],
            "lemmas": details["lemmas"],
            "overlap_count": overlap_count,
            "overlap_tokens": overlap_tokens,
            "jaccard_score": jaccard_score,
            "gloss_token_count": len(gloss_content_tokens),
            "details": details
        })

    # Sort candidates by overlap_count descending, then jaccard_score descending
    candidate_results.sort(key=lambda x: (x["overlap_count"], x["jaccard_score"]), reverse=True)

    # Assign rank numbers
    for idx, cand in enumerate(candidate_results, start=1):
        cand["rank"] = idx

    best_match = candidate_results[0]
    is_confident = best_match["overlap_count"] > 0

    return {
        "success": True,
        "target_word": cleaned_word,
        "context_sentence": context_sentence,
        "context_tokens": context_content_tokens,
        "context_lemmatized": sorted(list(context_lemmatized_set)),
        "total_candidates": len(candidate_results),
        "candidate_results": candidate_results,
        "best_match": best_match,
        "is_confident": is_confident,
        "methodology": {
            "algorithm": "Simplified Lesk (Token Overlap & Lemmatized Intersection)",
            "description": (
                "Computes the set intersection between lemmatized context content words "
                "(excluding the target word) and each candidate WordNet synset's gloss (definition + examples + synonyms). "
                "The synset with the maximum shared content vocabulary is ranked as the best contextual match."
            )
        }
    }
