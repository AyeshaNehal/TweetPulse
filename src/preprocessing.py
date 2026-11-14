# Owner: M. Hamdan (Labs 03, 04, 12)

import re
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# resource name -> path used by nltk.data.find
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "vader_lexicon": "sentiment/vader_lexicon.zip",
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}

# Negations are kept because they flip sentiment ("not good").
NEGATIONS = {"no", "nor", "not", "never", "none", "nobody", "nothing",
             "neither", "nowhere", "cannot"}

# Extra words to drop (tweet noise). Add new stopwords here.
EXTRA_STOPWORDS = {"amp"}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
REPEAT_PATTERN = re.compile(r"(.)\1{2,}")      # 3+ of the same letter in a row
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")

_SPECIAL_NEGATIONS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\bshan't\b", "shall not"),
    (r"\bain't\b", "is not"),
]

_stopword_cache = None


def ensure_nltk_resources():
    """Download any missing NLTK resources (only if they are not installed)."""
    for name, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def get_stopwords():
    """Return English stopwords as a set, keeping negations like 'not' and 'never'."""
    try:
        words = set(stopwords.words("english"))
    except LookupError:
        ensure_nltk_resources()
        words = set(stopwords.words("english"))
    return (words - NEGATIONS) | EXTRA_STOPWORDS


def _default_stopwords():
    """Load the stopword set once and reuse it (the corpus read is slow)."""
    global _stopword_cache
    if _stopword_cache is None:
        _stopword_cache = get_stopwords()
    return _stopword_cache


def normalise_quotes(text):
    """Replace backticks and curly quotes with a plain apostrophe."""
    return text.replace("`", "'").replace("\u2019", "'").replace("\u2018", "'")


def expand_negations(text):
    """Expand contractions: "didn't" -> "did not", "won't" -> "will not"."""
    for pattern, replacement in _SPECIAL_NEGATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return re.sub(r"\b(\w+)n't\b", r"\1 not", text, flags=re.IGNORECASE)


def clean_text(text, remove_urls=True, remove_mentions=True,
               collapse_repeats=True, remove_stopwords=True,
               min_length=2, stop_words=None):
    """Clean one tweet and return a list of tokens.

    Steps: normalise quotes, lowercase, remove URLs and @mentions, drop the
    '#' sign but keep the hashtag word, expand negations, shorten stretched
    words ('soooo' -> 'so'), keep letters only, then filter short tokens
    and stopwords.

    Args:
        text: the raw tweet (non-strings such as NaN give an empty list).
        remove_urls: strip http(s):// and www links.
        remove_mentions: strip @username.
        collapse_repeats: shorten 3+ repeated letters to one.
        remove_stopwords: drop stopwords (negations are kept).
        min_length: drop tokens shorter than this.
        stop_words: custom stopword set (default: get_stopwords()).
    """
    if not isinstance(text, str):
        return []

    text = normalise_quotes(text).lower()
    if remove_urls:
        text = URL_PATTERN.sub(" ", text)
    if remove_mentions:
        text = MENTION_PATTERN.sub(" ", text)
    text = text.replace("#", "")
    text = expand_negations(text)
    if collapse_repeats:
        text = REPEAT_PATTERN.sub(r"\1", text)
    text = NON_LETTER_PATTERN.sub(" ", text)

    tokens = [t for t in text.split() if t]
    stops = stop_words if stop_words is not None else _default_stopwords()
    tokens = list(filter(
        lambda t: len(t) >= min_length and not (remove_stopwords and t in stops),
        tokens))
    return tokens


def add_clean_column(df, text_col="text"):
    """Add 'tokens' (list of words) and 'clean_text' (joined string) columns."""
    df = df.copy()
    df["tokens"] = df[text_col].map(lambda t: clean_text(t))
    df["clean_text"] = df["tokens"].map(lambda toks: " ".join(toks))
    return df


def word_frequencies(token_lists):
    """Count words across many tweets. Returns a dict {word: count}."""
    freq = {}
    for tokens in token_lists:
        for word in tokens:
            freq[word] = freq.get(word, 0) + 1
    return freq


def top_words(freq, n=10):
    """Return the n most frequent words as a list of (word, count) tuples."""
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]


def build_vocabulary(token_lists, min_count=1):
    """Return the set of unique words that appear at least min_count times."""
    freq = word_frequencies(token_lists)
    return {word for word, count in freq.items() if count >= min_count}


def nltk_demo(text="I did not love this movie at all"):
    """Show NLTK tokens and POS tags for one sentence (Lab 12)."""
    ensure_nltk_resources()
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    print("Tokens:", tokens)
    print("POS tags:", tags)
    return tags


def demo_cleaning():
    """Print before/after examples for the report."""
    samples = [
        "Check this out https://t.co/abc123 @john #Happy day!",
        "soooo happy today!!! loooove it",
        "I didn`t like it, it wasn't good",
        "@mary I can't believe it... won't go again #fail",
        "The movie was NOT bad, never been so surprised",
    ]
    for s in samples:
        print("BEFORE:", s)
        print("AFTER :", clean_text(s))
        print()


if __name__ == "__main__":
    demo_cleaning()