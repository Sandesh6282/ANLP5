"""
WordNet Explorer Module for ANLP5.
Provides the primary interface for querying Princeton WordNet lexical database via NLTK.
Extracts synsets, pos distributions, lemma inventories, and taxonomy metrics.
"""

from typing import List, Dict, Any, Optional
from nltk.corpus import wordnet as wn
from utils.helpers import clean_word_input, format_lemma_name, get_pos_label


def get_synsets(word: str, pos: Optional[str] = None) -> List[Any]:
    """
    Safely retrieves WordNet synsets for a given word and optional POS filter.
    Returns an empty list if not found or on error.
    """
    cleaned = clean_word_input(word)
    if not cleaned:
        return []
    try:
        if pos:
            return wn.synsets(cleaned, pos=pos)
        return wn.synsets(cleaned)
    except Exception:
        return []


def word_exists_in_wordnet(word: str) -> bool:
    """Checks whether the word has at least one synset entry in WordNet."""
    synsets = get_synsets(word)
    return len(synsets) > 0


def get_synset_details(synset) -> Dict[str, Any]:
    """
    Extracts rich, structured metadata for a single WordNet Synset.
    """
    raw_lemmas = [lemma.name() for lemma in synset.lemmas()]
    formatted_lemmas = []
    for l in raw_lemmas:
        clean_l = format_lemma_name(l)
        if clean_l not in formatted_lemmas:
            formatted_lemmas.append(clean_l)
            
    # Hypernyms (broader terms)
    hypernyms = []
    try:
        hypernyms = [h.name().split('.')[0].replace('_', ' ') for h in synset.hypernyms()]
    except Exception:
        pass

    # Hyponym count (narrower terms)
    hyponym_count = 0
    try:
        hyponym_count = len(synset.hyponyms())
    except Exception:
        pass

    # Taxonomy depth
    try:
        min_depth = synset.min_depth()
        max_depth = synset.max_depth()
    except Exception:
        min_depth = 0
        max_depth = 0

    return {
        "synset": synset,
        "name": synset.name(),
        "pos": synset.pos(),
        "pos_label": get_pos_label(synset.pos()),
        "definition": synset.definition(),
        "examples": synset.examples(),
        "raw_lemmas": raw_lemmas,
        "lemmas": formatted_lemmas,
        "hypernyms": hypernyms,
        "hyponym_count": hyponym_count,
        "min_depth": min_depth,
        "max_depth": max_depth
    }


def get_word_overview(word: str) -> Dict[str, Any]:
    """
    Aggregates comprehensive lexical statistics for a searched word.
    """
    cleaned = clean_word_input(word)
    synsets = get_synsets(cleaned)

    if not synsets:
        return {
            "word": cleaned,
            "found": False,
            "total_synsets": 0,
            "pos_counts": {},
            "unique_synonyms": [],
            "unique_synonyms_count": 0,
            "total_examples": 0,
            "synset_details": []
        }

    synset_details = [get_synset_details(s) for s in synsets]
    
    # Calculate POS distribution
    pos_counts = {}
    for item in synset_details:
        pos_code = item["pos"]
        pos_counts[pos_code] = pos_counts.get(pos_code, 0) + 1

    # Extract all unique synonyms across all senses (excluding the word itself or case variations)
    all_synonyms_set = set()
    total_examples = 0

    for item in synset_details:
        total_examples += len(item["examples"])
        for lemma in item["lemmas"]:
            # Standardize and add
            if lemma.lower() != cleaned.lower():
                all_synonyms_set.add(lemma)

    unique_synonyms = sorted(list(all_synonyms_set))

    return {
        "word": cleaned,
        "found": True,
        "total_synsets": len(synsets),
        "pos_counts": pos_counts,
        "unique_synonyms": unique_synonyms,
        "unique_synonyms_count": len(unique_synonyms),
        "total_examples": total_examples,
        "synset_details": synset_details,
        "primary_sense": synset_details[0] if synset_details else None
    }
