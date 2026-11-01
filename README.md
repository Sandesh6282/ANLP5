# ANLP5 – Synonym and Word Sense Identification using NLTK WordNet

**Course:** Advanced Natural Language Processing (AML23702)  
**Evaluation:** CIA Laboratory Exercise – Question 5  
**Project Title:** ANLP5 WordNet Explorer  

---

## 1. Project Overview

**ANLP5 WordNet Explorer** is a modern, interactive web application developed for exploring word senses, lexical synonyms, semantic hierarchies, and contextual word sense disambiguation using the Princeton WordNet lexical database via the Python Natural Language Toolkit (NLTK).

The application functions as a laboratory dashboard designed for academic demonstration and research. It retrieves live, authentic data from the WordNet database rather than relying on static or mocked lexical entries.

---

## 2. Problem Statement

> **Problem Statement:**  
> Write a Python program using NLTK WordNet to identify the synonyms and definitions of a given word and distinguish its possible meanings in different contexts.

### Key Computational Challenges Addressed:
1. **Lexical Polysemy & Homonymy:** A single word token (such as *"bank"*, *"light"*, or *"bat"*) maps to multiple distinct semantic concepts depending on its syntactic category and contextual framing.
2. **Sense-Based Synonym Grouping:** Standard synonym lookups often produce flat, unstructured lists that conflate distinct meanings. In contrast, WordNet groups synonyms under discrete cognitive synonym sets (*synsets*).
3. **Word Sense Disambiguation (WSD):** Distinguishing the intended meaning of an ambiguous word in an arbitrary sentence by comparing surrounding content words with dictionary glosses using explainable algorithms (Simplified Lesk).

---

## 3. Objectives

- **Synset Identification:** Retrieve and inspect all cognitive synonym sets (synsets) associated with any English word across Nouns, Verbs, Adjectives, and Adverbs.
- **Definition & Gloss Extraction:** Display formal lexicographic definitions and corpus-grounded example sentences.
- **Grouped Synonym Analysis:** Group synonyms by their specific WordNet sense to prevent semantic confusion.
- **Explainable Word Sense Disambiguation:** Provide an educational case study contrasting the senses of *"bank"* (e.g., Financial institution vs. River slope) and an interactive interface evaluating custom user sentences using the Simplified Lesk algorithm.
- **Interactive Visualizations:** Generate publication-ready Plotly charts illustrating part-of-speech distributions, lexical richness, WordNet taxonomy depth, and sunburst hierarchies.

---

## 4. Key Modules & Features

| Module | Title | Description |
| :--- | :--- | :--- |
| **Module 1** | **Dashboard / Overview** | Project metadata, problem statement, technology stack, WordNet structural guide, and operational instructions. |
| **Module 2** | **Word Search** | Real-time search engine with input validation, WordNet availability verification, key metric cards, and roster table. |
| **Module 3** | **Synonym Explorer** | Synonyms grouped strictly by synset meaning, deduplicated lemmas, and cross-sense unique synonym cloud. |
| **Module 4** | **Definition Explorer** | Granular inspection of synset definitions, example sentences, lemmas, direct hypernyms (super-concepts), and depth. |
| **Module 5** | **Word Sense Disambiguation** | Foundational *"bank"* case study comparing Financial vs. River vs. Flight senses, and interactive dual-sense showcases. |
| **Module 6** | **Contextual Meaning Analysis** | Transparent Simplified Lesk algorithm with tokenization breakdown, lemmatized matching, overlap scores, and candidate rankings. |
| **Module 7** | **WordNet Visualizations** | Plotly charts: POS Donut, Synonyms per Sense Bar, Taxonomy Tree Depth Bar, and Lexical Sunburst Tree. |
| **Module 8** | **Sample Demonstrations** | One-click benchmark loading for `bank`, `book`, `apple`, `light`, `bat`, `run`, and `computer`. |

---

## 5. Technology Stack

- **Programming Language:** Python (>= 3.10, tested on 3.13)
- **Web Application Framework:** [Streamlit](https://streamlit.io/)
- **Natural Language Toolkit:** [NLTK](https://www.nltk.org/)
- **Lexical Semantic Database:** [Princeton WordNet 3.0](https://wordnet.princeton.edu/) & Open Multilingual Wordnet (`omw-1.4`)
- **Data Structuring:** [Pandas](https://pandas.pydata.org/)
- **Data Visualization:** [Plotly](https://plotly.com/python/)
- **Interface Styling:** Custom CSS (Inter & Outfit typography, glassmorphism, responsive cards)

---

## 6. Project Folder Structure

```
ANLP5/
├── app.py                      # Main Streamlit web application & routing
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── assets/
│   └── style.css               # Modern academic dashboard stylesheet
├── modules/
│   ├── __init__.py
│   ├── wordnet_explorer.py     # Core WordNet querying and overview statistics
│   ├── synonym_extractor.py    # Sense-grouped synonym extraction & deduplication
│   ├── definition_extractor.py # Synset definition cards and hierarchy extraction
│   ├── word_sense.py           # Word Sense Disambiguation case studies (bank)
│   ├── context_analyzer.py     # Simplified Lesk algorithm for contextual disambiguation
│   └── visualizations.py       # Interactive Plotly charts based on real WordNet data
└── utils/
    ├── __init__.py
    ├── nltk_setup.py           # Automated NLTK corpus verification & SSL handler
    └── helpers.py              # POS mapping, text cleaning, and tokenization utilities
```

---

## 7. WordNet Implementation Details

The application interfaces directly with NLTK's `nltk.corpus.wordnet` API:

```python
from nltk.corpus import wordnet as wn

# 1. Retrieve all synsets for a query word
synsets = wn.synsets('bank')

# 2. Extract properties for each synset
for s in synsets:
    synset_name = s.name()          # e.g., 'bank.n.01'
    pos_tag = s.pos()               # 'n', 'v', 'a', 'r', 's'
    definition = s.definition()     # Definition string
    examples = s.examples()         # List of example sentences
    lemmas = [l.name() for l in s.lemmas()]  # Synonyms within this sense
    hypernyms = s.hypernyms()       # Broader category synsets
    min_depth = s.min_depth()       # Distance to root entity.n.01
```

### Contextual Disambiguation Logic (Simplified Lesk)

WordNet itself acts as a **static sense repository**; it does not automatically classify arbitrary sentences. The application bridges this gap with an explainable Simplified Lesk algorithm in `modules/context_analyzer.py`:

$$\text{Overlap}(S, C) = |\text{Lemmas}(\text{ContextContentWords}) \cap \text{Lemmas}(\text{SynsetGloss})|$$

Where:
- $\text{ContextContentWords}$ = Sentence tokens with stopwords, punctuation, and the target word removed.
- $\text{SynsetGloss}$ = Tokens from the candidate synset's definition, examples, and lemmas.
- $\text{Score}$ = Shared token count and Jaccard similarity index.
- Fallback = If overlap is zero, the algorithm transparently defaults to WordNet's Most Frequent Sense (MFS) rather than fabricating numbers.

---

## 8. Installation & Setup Instructions

### Step 1: Clone or Navigate to the Workspace
```bash
cd /Users/sandeshj/ANLP5
```

### Step 2: Create and Activate a Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify NLTK Corpora (Automated)
The application automatically checks and downloads missing resources on launch (`wordnet`, `omw-1.4`, `punkt`, `punkt_tab`, `stopwords`). If manual download is preferred:
```bash
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

## 9. Launching the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

Or run via the virtual environment path:
```bash
.venv/bin/streamlit run app.py
```

### Local Application URL:
Once started, the dashboard is accessible at:
- **Local URL:** `http://localhost:8501`
- **Network URL:** Provided by Streamlit in the terminal output.

---

## 10. Verification Instructions for the Evaluator

1. **Dashboard:** Open the application to inspect the problem statement, objectives, and technology stack.
2. **Word Search:** Enter `"bank"` and click **Search**. Verify the 4 metric cards (18 Synsets, 13 Unique Synonyms, 2 Parts of Speech).
3. **Synonym Explorer:** Inspect how synonyms like `depository_financial_institution`, `banking_company`, and `bank` are grouped strictly under the financial sense, distinct from the slope sense.
4. **Definition Explorer:** Filter definitions by keyword (e.g. `"water"` or `"money"`) and review synset IDs, definitions, and hypernyms.
5. **Word Sense Disambiguation:** Review the side-by-side contrast of `bank.n.01` (River slope) and `depository_financial_institution.n.01` (Financial institution).
6. **Contextual Meaning Analysis:** Test the preset sentence *"The river bank was muddy after the heavy rainfall"* and observe how `bank.n.01` is ranked #1 with overlapping clue word `river`.
7. **Visualizations:** Navigate to Module 7 and interact with the POS Donut chart, Synonyms per Sense Bar chart, Taxonomy Depth chart, and the interactive Lexical Sunburst.
8. **Sample Words:** Test the benchmark words (`light`, `bat`, `apple`, `book`, `run`, `computer`) to verify dynamic updates.
