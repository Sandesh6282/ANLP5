"""
Definition Explorer Module for ANLP5.
Retrieves and structures all available WordNet meanings, glosses, and grammatical roles.
Formats synset details into clean, structured data for separate card presentations.
"""

from typing import List, Dict, Any, Optional
from modules.wordnet_explorer import get_synsets, get_synset_details
from utils.helpers import clean_word_input


def extract_definitions(word: str, pos_filter: Optional[str] = None, search_term: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extracts all available WordNet definitions and metadata for the given word.
    Supports filtering by POS and matching keywords within definitions.
    """
    cleaned = clean_word_input(word)
    synsets = get_synsets(cleaned, pos=pos_filter)
    
    definitions = []
    
    for idx, synset in enumerate(synsets, start=1):
        details = get_synset_details(synset)
        
        # If user supplied a sub-search keyword inside definitions
        if search_term:
            st = search_term.strip().lower()
            in_def = st in details["definition"].lower()
            in_ex = any(st in ex.lower() for ex in details["examples"])
            in_lemmas = any(st in lm.lower() for lm in details["lemmas"])
            if not (in_def or in_ex or in_lemmas):
                continue

        definitions.append({
            "index": idx,
            "synset_id": synset.name(),
            "pos": details["pos"],
            "pos_label": details["pos_label"],
            "definition": details["definition"],
            "examples": details["examples"],
            "lemmas": details["lemmas"],
            "hypernyms": details["hypernyms"],
            "hyponym_count": details["hyponym_count"],
            "min_depth": details["min_depth"],
            "max_depth": details["max_depth"]
        })
        
    return definitions
