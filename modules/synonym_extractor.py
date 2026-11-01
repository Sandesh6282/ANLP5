"""
Synonym Extractor Module for ANLP5.
Extracts, structures, and groups synonyms by corresponding WordNet senses (synsets).
Ensures deduplication, meaning-based categorization, and part-of-speech filtering.
"""

from typing import List, Dict, Any, Optional
from modules.wordnet_explorer import get_synsets, get_synset_details
from utils.helpers import clean_word_input, format_lemma_name


def extract_synonyms_by_meaning(word: str, pos_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extracts synonyms grouped by their specific WordNet synset/meaning.
    
    Each meaning group contains:
    - synset_name: e.g. depository_financial_institution.n.01
    - pos: e.g. 'n'
    - pos_label: e.g. 'Noun'
    - definition: synset definition string
    - synonyms: list of unique synonyms (lemmas) for this specific sense
    - examples: example sentences
    """
    cleaned = clean_word_input(word)
    synsets = get_synsets(cleaned, pos=pos_filter)
    
    meaning_groups = []
    
    for synset in synsets:
        details = get_synset_details(synset)
        
        # Deduplicate and clean synonyms for this synset
        raw_lemmas = [lemma.name() for lemma in synset.lemmas()]
        synonyms = []
        for l in raw_lemmas:
            clean_l = format_lemma_name(l)
            # We preserve all lemmas in the synset (including or highlighting the query word)
            if clean_l and clean_l not in synonyms:
                synonyms.append(clean_l)
                
        meaning_groups.append({
            "synset_name": synset.name(),
            "pos": synset.pos(),
            "pos_label": details["pos_label"],
            "definition": synset.definition(),
            "synonyms": synonyms,
            "target_word": cleaned,
            "examples": synset.examples(),
            "hypernyms": details["hypernyms"]
        })
        
    return meaning_groups


def extract_all_unique_synonyms(word: str) -> List[str]:
    """
    Returns an aggregated, deduplicated list of all synonyms across all senses of the word.
    """
    cleaned = clean_word_input(word)
    synsets = get_synsets(cleaned)
    unique_set = set()
    
    for s in synsets:
        for lemma in s.lemmas():
            fmt = format_lemma_name(lemma.name())
            if fmt.lower() != cleaned.lower():
                unique_set.add(fmt)
                
    return sorted(list(unique_set))
