"""
NLTK Resource Setup & Corpus Management for ANLP5 WordNet Explorer.
Ensures WordNet, OMW-1.4, tokenizers, and stopwords are verified and available.
Handles SSL certificate checks and provides safe offline fallbacks.
"""

import os
import ssl
import logging
import nltk

logger = logging.getLogger(__name__)

# Required NLTK resources for WordNet Explorer and Contextual Disambiguation
REQUIRED_RESOURCES = [
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]


def ensure_nltk_resources() -> dict:
    """
    Checks and ensures all necessary NLTK corpora are downloaded.
    Returns a status dictionary detailing downloaded resources and any errors.
    """
    # Fix SSL certificate verification issues common in macOS Python builds
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    results = {
        "success": True,
        "available": [],
        "downloaded": [],
        "failed": [],
        "errors": []
    }

    for path, package in REQUIRED_RESOURCES:
        try:
            # Check if resource already exists
            nltk.data.find(path)
            results["available"].append(package)
        except LookupError:
            try:
                logger.info(f"Downloading required NLTK package: {package}")
                nltk.download(package, quiet=True)
                results["downloaded"].append(package)
            except Exception as e:
                logger.warning(f"Failed to download NLTK package {package}: {e}")
                results["failed"].append(package)
                results["errors"].append(str(e))
                # Only wordnet is strictly critical; others have fallbacks
                if package == "wordnet":
                    results["success"] = False

    return results
