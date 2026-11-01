"""
Helper functions and constants for ANLP5 WordNet Explorer.
Provides POS mapping, text cleaning, tokenization, and formatting utilities.
"""

import re
import string
from typing import Set, List

POS_MAP = {
    'n': 'Noun',
    'v': 'Verb',
    'a': 'Adjective',
    's': 'Adjective Satellite',
    'r': 'Adverb'
}

POS_COLORS = {
    'n': '#2563EB',  # Blue
    'v': '#059669',  # Emerald
    'a': '#D97706',  # Amber
    's': '#7C3AED',  # Purple
    'r': '#DB2777'   # Pink
}

POS_BADGES = {
    'n': 'badge-noun',
    'v': 'badge-verb',
    'a': 'badge-adj',
    's': 'badge-adjsat',
    'r': 'badge-adv'
}

POS_ICONS = {
    'n': '🏷️',
    'v': '⚡',
    'a': '✨',
    's': '🌟',
    'r': '🎯'
}

# Standard English stopwords fallback in case NLTK corpus has any delay
DEFAULT_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t',
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have',
    'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into',
    'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my',
    'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s',
    'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re',
    'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t',
    'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s',
    'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t',
    'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself',
    'yourselves'
}


def get_stopwords() -> Set[str]:
    """Retrieve NLTK stopwords or fallback to standard English stopwords."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english')).union(DEFAULT_STOPWORDS)
    except Exception:
        return DEFAULT_STOPWORDS


def format_lemma_name(raw_lemma: str) -> str:
    """Formats lemma string by replacing underscores with spaces and capitalizing properly."""
    if not raw_lemma:
        return ""
    return raw_lemma.replace('_', ' ').strip()


def get_pos_label(pos_code: str) -> str:
    """Returns human-friendly part-of-speech string."""
    return POS_MAP.get(pos_code.lower(), f"Unknown ({pos_code})")


def get_pos_color(pos_code: str) -> str:
    """Returns hex color code for part-of-speech badge."""
    return POS_COLORS.get(pos_code.lower(), '#6B7280')


def clean_word_input(text: str) -> str:
    """Cleans user search input: removes outer punctuation/spaces and lowercases."""
    if not text:
        return ""
    # Strip whitespace and common surrounding quotes/brackets
    cleaned = text.strip().strip("'\"`.,;:!?-")
    return cleaned.lower()


def is_valid_word_query(word: str) -> tuple[bool, str]:
    """
    Validates word input string.
    Returns (is_valid, error_message).
    """
    if not word or not word.strip():
        return False, "Please enter a word to search."
    
    cleaned = clean_word_input(word)
    if len(cleaned) < 1:
        return False, "Word cannot be empty."
    
    if len(cleaned) > 40:
        return False, "Search query is too long. Please enter a valid single word or phrase."
    
    # Allow letters, spaces, hyphens (for compound words like 'ice cream', 'well-known')
    if not re.match(r'^[a-zA-Z\s\-]+$', cleaned):
        return False, "Query contains invalid characters. Please enter alphabetic words."
    
    return True, ""


def tokenize_and_filter(text: str) -> List[str]:
    """
    Tokenizes text, strips punctuation, normalizes to lowercase,
    and removes common stopwords for content-word overlap comparison.
    """
    if not text:
        return []
    
    stops = get_stopwords()
    # Extract purely alphanumeric words
    tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    content_tokens = [tok for tok in tokens if tok not in stops]
    return content_tokens
