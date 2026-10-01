"""
======================================================================
HATE SPEECH AND OFFENSIVE LANGUAGE DETECTION — TEXT PREPROCESSING
======================================================================
Academic NLP Text Preprocessing Module for Hate Speech Detection.
======================================================================
"""

import math
import re
import string
from typing import Dict, Iterable, List, Optional, Union

# ====================================================================
# CONFIGURATION CONSTANTS & PRECOMPILED REGULAR EXPRESSIONS
# ====================================================================
# Precompiling regex patterns avoids redundant re-compilation per text string,
# significantly improving performance across large datasets.

# Pattern matching HTTP, HTTPS, and WWW URLs
URL_PATTERN: re.Pattern = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)

# Pattern matching Twitter/X style user mentions (e.g. @username)
MENTION_PATTERN: re.Pattern = re.compile(r"@\w+", re.IGNORECASE)

# Pattern matching HTML tags (e.g. <p>, <br>, <div>)
HTML_TAG_PATTERN: re.Pattern = re.compile(r"<.*?>", re.IGNORECASE)

# Pattern matching HTML character entities (e.g. &amp;, &lt;, &gt;)
HTML_ENTITY_PATTERN: re.Pattern = re.compile(r"&[a-z0-9]+;", re.IGNORECASE)

# Pattern matching multiple consecutive whitespace characters
WHITESPACE_PATTERN: re.Pattern = re.compile(r"\s+")

# Translation table for efficient punctuation removal using str.translate()
PUNCTUATION_TABLE: Dict[int, Optional[int]] = str.maketrans("", "", string.punctuation)


# ====================================================================
# INPUT NORMALIZATION & TYPE HANDLING
# ====================================================================
def normalize_input_type(text: Optional[Union[str, float, int]]) -> str:
    """
    Safely handle and normalize arbitrary input types into string form.

    Parameters
    ----------
    text : Optional[Union[str, float, int]]
        Input text or scalar value.

    Returns
    -------
    str
        Validated string representation or empty string for missing values.

    Notes
    -----
    Null or NaN values are safely converted to empty strings "" rather than
    literal strings like "nan" or "None" to prevent data corruption in TF-IDF.
    """
    if text is None:
        return ""

    if isinstance(text, float):
        if math.isnan(text):
            return ""
        return str(text)

    if not isinstance(text, str):
        return str(text)

    return text


# ====================================================================
# INDIVIDUAL TEXT CLEANING PIPELINE STAGES
# ====================================================================
def lowercase_text(text: str) -> str:
    """
    Convert text string to lowercase.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Lowercased text string.
    """
    return text.lower()


def remove_urls(text: str) -> str:
    """
    Remove HTTP, HTTPS, and WWW URL links from text.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Text with all URLs stripped.
    """
    return URL_PATTERN.sub("", text)


def remove_mentions(text: str) -> str:
    """
    Remove user mention handles (@username) from text.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Text with all user mentions removed.
    """
    return MENTION_PATTERN.sub("", text)


def remove_html_tags(text: str) -> str:
    """
    Remove HTML markup tags and character entities from text.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Text with HTML tags and entities removed.
    """
    text = HTML_TAG_PATTERN.sub("", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)
    return text


def remove_punctuation(text: str) -> str:
    """
    Remove standard ASCII punctuation marks from text string.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Text string with punctuation symbols removed.
    """
    return text.translate(PUNCTUATION_TABLE)


def normalize_special_characters(text: str) -> str:
    """
    Normalize or strip unwanted non-alphanumeric noise characters conservatively.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Normalized text string.
    """
    # Replace non-printable ASCII control characters if any exist
    return "".join(ch for ch in text if ch.isprintable())


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple consecutive whitespace characters and trim edges.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Text string with single-space separation and stripped edges.
    """
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def is_empty_text(text: str) -> bool:
    """
    Check if a text string is empty or contains only whitespace characters.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    bool
        True if the text is empty after stripping, False otherwise.
    """
    return len(text.strip()) == 0


# ====================================================================
# MAIN PUBLIC API FUNCTION: clean_text
# ====================================================================
def clean_text(text: Optional[Union[str, float, int]]) -> str:
    """
    Main public text preprocessing pipeline function.

    Executes a sequential sequence of text cleaning operations:
      1. Normalize input type (handles None, NaN, numeric types safely)
      2. Lowercase text
      3. Remove URLs (http, https, www)
      4. Remove @user mentions
      5. Remove HTML tags and entities
      6. Remove ASCII punctuation
      7. Normalize special non-printable characters
      8. Normalize whitespace and collapse extra spaces

    Parameters
    ----------
    text : Optional[Union[str, float, int]]
        Raw input string or scalar.

    Returns
    -------
    str
        Normalized, clean text string ready for TF-IDF feature extraction.
    """
    text = normalize_input_type(text)

    if not text:
        return ""

    text = lowercase_text(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = normalize_special_characters(text)
    text = normalize_whitespace(text)

    return text


# ====================================================================
# BATCH & UTILITY FUNCTIONS
# ====================================================================
def clean_texts(texts: Iterable[Optional[Union[str, float, int]]]) -> List[str]:
    """
    Apply clean_text preprocessing to an iterable of raw text inputs.

    Parameters
    ----------
    texts : Iterable[Optional[Union[str, float, int]]]
        Iterable collection of raw input texts.

    Returns
    -------
    List[str]
        List of preprocessed text strings.
    """
    return [clean_text(t) for t in texts]


def get_text_statistics(text: str) -> Dict[str, Union[int, bool]]:
    """
    Calculate basic text length and word count statistics.

    Parameters
    ----------
    text : str
        Input raw or cleaned text string.

    Returns
    -------
    Dict[str, Union[int, bool]]
        Dictionary containing character length, word count, and empty flag.
    """
    normalized = normalize_input_type(text)
    words = normalized.split() if normalized else []

    return {
        "character_count": len(normalized),
        "word_count": len(words),
        "is_empty": len(words) == 0,
    }


def compare_text_versions(text: str) -> Dict[str, Union[str, int, bool]]:
    """
    Compare raw input text against its preprocessed clean version.

    Parameters
    ----------
    text : str
        Raw input string.

    Returns
    -------
    Dict[str, Union[str, int, bool]]
        Comparative dictionary containing original vs cleaned text metrics.
    """
    cleaned = clean_text(text)
    orig_stats = get_text_statistics(text)
    clean_stats = get_text_statistics(cleaned)

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "original_length": orig_stats["character_count"],
        "cleaned_length": clean_stats["character_count"],
        "original_word_count": orig_stats["word_count"],
        "cleaned_word_count": clean_stats["word_count"],
    }


def preview_cleaning(text: str) -> None:
    """
    Print a safe development preview comparison of raw text vs cleaned text.

    Parameters
    ----------
    text : str
        Input raw text string.
    """
    comp = compare_text_versions(text)
    print("=" * 60)
    print("TEXT PREPROCESSING PREVIEW")
    print("=" * 60)
    print(f"Original Text        : {comp['original_text']}")
    print(f"Cleaned Text         : {comp['cleaned_text']}")
    print(f"Original Character Len: {comp['original_length']}")
    print(f"Cleaned Character Len : {comp['cleaned_length']}")
    print(f"Original Word Count  : {comp['original_word_count']}")
    print(f"Cleaned Word Count   : {comp['cleaned_word_count']}")
    print("=" * 60)


# ====================================================================
# OPTIONAL STANDALONE MODULE TEST
# ====================================================================
if __name__ == "__main__":
    print("Running preprocessing module self-tests...")

    sample_test_sentences = [
        "HELLO WORLD",
        "Visit https://example.com today!",
        "Hello @username welcome to the project.",
        "<p>Hello World</p>",
        "Hello, World!!!",
        "   Hello       World   ",
        None,
    ]

    for sample in sample_test_sentences:
        result = clean_text(sample)
        print(f"Input: {repr(sample):45s} -> Cleaned: {repr(result)}")

    print("\nSelf-tests completed successfully.")
