"""
Word Sense Disambiguation (WSD) Educational Module for ANLP5.
Demonstrates lexical polysemy and homonymy using NLTK WordNet.
Features the classic 'bank' case study (financial institution vs river slope),
explains the difference between WordNet sense inventory and contextual disambiguation,
and provides comparative sense breakdowns.
"""

from typing import Dict, List, Any
from nltk.corpus import wordnet as wn
from modules.wordnet_explorer import get_synset_details


def get_bank_case_study() -> Dict[str, Any]:
    """
    Retrieves the foundational WordNet synsets and contrastive details for 'bank'.
    Demonstrates:
    1. Financial Bank (depository_financial_institution.n.01)
    2. River Bank (bank.n.01)
    3. Additional senses (Flight maneuver, Reserve supply, Verb senses)
    """
    # River bank synset
    river_synset = wn.synset('bank.n.01')
    # Financial institution synset
    financial_synset = wn.synset('depository_financial_institution.n.01')
    # Flight maneuver synset
    flight_synset = wn.synset('bank.n.10')
    # Financial building synset
    building_synset = wn.synset('bank.n.09')

    return {
        "word": "bank",
        "primary_contrast": {
            "sense_1": {
                "title": "River Bank (Geographical Slope)",
                "context_example": "The river bank was muddy after days of torrential rainfall.",
                "details": get_synset_details(river_synset),
                "key_discriminators": ["river", "water", "slope", "muddy", "shore", "stream", "land"]
            },
            "sense_2": {
                "title": "Financial Bank (Monetary Institution)",
                "context_example": "I deposited my paycheck into the bank and checked my savings account.",
                "details": get_synset_details(financial_synset),
                "key_discriminators": ["money", "deposit", "account", "institution", "finance", "loan", "check", "paycheck"]
            }
        },
        "additional_senses": [
            {
                "title": "Flight Maneuver (Aeronautics)",
                "context_example": "The pilot executed a steep bank to the left before aligning with the runway.",
                "details": get_synset_details(flight_synset),
                "key_discriminators": ["aircraft", "pilot", "turn", "lateral", "flight", "steep"]
            },
            {
                "title": "Banking Building (Physical Architecture)",
                "context_example": "He walked into the bank on the corner of 5th Avenue.",
                "details": get_synset_details(building_synset),
                "key_discriminators": ["building", "corner", "street", "architecture", "office"]
            }
        ]
    }


POLYSEMOUS_SHOWCASES = {
    "bat": {
        "description": "Zoological creature vs. Sports equipment",
        "sense_a": {
            "synset_id": "bat.n.01",
            "title": "Zoological Creature (Mammal)",
            "sentence": "The nocturnal bat flew out of the dark cave at dusk.",
            "keywords": ["cave", "flying", "nocturnal", "wings", "mammal", "dusk"]
        },
        "sense_b": {
            "synset_id": "bat.n.05",
            "title": "Sports Equipment (Club)",
            "sentence": "The cricket batsman swung the wooden bat and struck the leather ball.",
            "keywords": ["wooden", "baseball", "cricket", "swing", "ball", "hit", "strike"]
        }
    },
    "apple": {
        "description": "Edible orchard fruit vs. Plant/Tree",
        "sense_a": {
            "synset_id": "apple.n.01",
            "title": "Edible Orchard Fruit",
            "sentence": "She sliced a crisp red apple for a healthy afternoon snack.",
            "keywords": ["fruit", "red", "crisp", "eat", "slice", "sweet", "orchard"]
        },
        "sense_b": {
            "synset_id": "apple.n.02",
            "title": "Apple Tree (Malus)",
            "sentence": "The farmer planted twenty apple trees in the newly cleared orchard.",
            "keywords": ["tree", "plant", "grow", "orchard", "cultivated", "malus"]
        }
    },
    "light": {
        "description": "Electromagnetic Radiation vs. Weight Adjective",
        "sense_a": {
            "synset_id": "light.n.01",
            "title": "Electromagnetic Illumination (Noun)",
            "sentence": "The bright sunlight flooded the room through the opened blinds.",
            "keywords": ["sunlight", "bright", "illumination", "darkness", "lamp", "beam"]
        },
        "sense_b": {
            "synset_id": "light.a.01",
            "title": "Low Weight (Adjective)",
            "sentence": "The carbon fiber backpack is surprisingly light and easy to carry on long hikes.",
            "keywords": ["weight", "heavy", "carry", "feather", "weigh", "pound", "easy"]
        }
    },
    "book": {
        "description": "Printed Publication (Noun) vs. Make Reservation (Verb)",
        "sense_a": {
            "synset_id": "book.n.01",
            "title": "Bound Publication (Noun)",
            "sentence": "She read a thrilling mystery book before going to sleep.",
            "keywords": ["read", "pages", "author", "chapters", "novel", "literature"]
        },
        "sense_b": {
            "synset_id": "book.v.01",
            "title": "Make Reservation (Verb)",
            "sentence": "We need to book a hotel room and flight tickets for our summer vacation.",
            "keywords": ["reserve", "flight", "tickets", "hotel", "schedule", "vacation"]
        }
    }
}


def get_showcase_comparison(word: str) -> Dict[str, Any]:
    """Retrieves showcase polysemy pair for predefined words."""
    if word.lower() in POLYSEMOUS_SHOWCASES:
        data = POLYSEMOUS_SHOWCASES[word.lower()]
        try:
            syn_a = wn.synset(data["sense_a"]["synset_id"])
            syn_b = wn.synset(data["sense_b"]["synset_id"])
            return {
                "word": word.lower(),
                "description": data["description"],
                "sense_a": {
                    "title": data["sense_a"]["title"],
                    "sentence": data["sense_a"]["sentence"],
                    "details": get_synset_details(syn_a),
                    "keywords": data["sense_a"]["keywords"]
                },
                "sense_b": {
                    "title": data["sense_b"]["title"],
                    "sentence": data["sense_b"]["sentence"],
                    "details": get_synset_details(syn_b),
                    "keywords": data["sense_b"]["keywords"]
                }
            }
        except Exception:
            return {}
    return {}
