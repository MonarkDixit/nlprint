# src/preprocessor.py
# NLPrint: Amazon Review Authenticity Detector
# Author: Monark Dixit (UID: 122259645)
# Course: MSML606, Spring 2026

#PURPOSE:
#This module is the first stage of the NLPrint pipeline. Every review passes through here before entering the hashing stage. 
#The preprocessor is responsible for:

#1. Stripping HTML artifacts (e.g. &amp; <br> <b>bold</b>)
#2. Normalizing unicode characters (accented letters, fancy quotes, etc.)
#3. Normalizing and tokenizing emoji into text tokens
#4. Lowercasing and removing punctuation
#5. Removing English stopwords (common words like "the", "is", "a")
#6. Detecting non-English reviews via character distribution (flag only)
#7. Tokenizing the cleaned text into overlapping character n-grams (also called "shingles") which are the core input to MinHash

#A shingle is a contiguous sequence of characters of length SHINGLE_SIZE.
#For example, with SHINGLE_SIZE=3 the word "hello" produces:
#       {"hel", "ell", "llo"}
#Overlapping shingles capture partial matches between near-duplicate reviews
#much better than word-level tokens do.

#USAGE:
#Import the preprocess() function in other modules:
    #from src.preprocessor import preprocess
    #result = preprocess("This product is amazing! <br> Great smell.")

#Run directly to validate output on 10 sample reviews:
#python src/preprocessor.py

import re
import unicodedata
import html
import string

#nltk is used for stopword removal.
#We download the stopwords corpus on first run if it is not already present.
import nltk

#langdetect is used for non-English detection via character distribution. It is lightweight and does not require a network call after install.
from langdetect import detect, LangDetectException


#NLTK SETUP
#Download the stopwords corpus if it is not already on disk. This only runs once; subsequent calls are instant.

def _ensure_nltk_data():        #Download required NLTK data silently if not already present.
    
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("[INFO] Downloading NLTK stopwords corpus...")
        nltk.download("stopwords", quiet=True)

_ensure_nltk_data()

from nltk.corpus import stopwords

#Build a set of English stopwords for O(1) lookup during preprocessing.
#Using a set instead of a list makes the "is this a stopword?" check
#constant time regardless of how many stopwords there are.
ENGLISH_STOPWORDS = set(stopwords.words("english"))


#CONSTANTS
#These values are tunable and affect the quality of the similarity search.

#SHINGLE_SIZE: The length of each character n-gram. Size 3 (trigrams) is a good balance: small enough to catch partial matches, large enough to avoid
#matching completely unrelated text. The proposal specifies character n-grams.

#MIN_SHINGLES: Reviews that produce fewer shingles than this after cleaning are considered too short to be meaningful and are flagged.

SHINGLE_SIZE = 3       #Character n-gram length (trigrams)
MIN_SHINGLES = 10      #Minimum shingle count for a review to be usable


#HTML STRIPPING
#Amazon reviews sometimes contain residual HTML from copy-paste or the review submission form. We need to remove tags and decode entities before any text processing happens.

#Regex pattern that matches any HTML tag, e.g. <br>, <b>, </div>
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:       #Removes HTML tags and decodes HTML entities from review text.
    if not text:
        return ""

    #Step 1: Decode HTML entities first so &lt;b&gt; becomes <b> before
    #the tag-stripping regex runs
    text = html.unescape(text)

    #Step 2: Replace HTML tags with a space rather than empty string
    #so words on either side of a tag do not accidentally merge
    text = _HTML_TAG_PATTERN.sub(" ", text)

    return text


#UNICODE NORMALIZATION
#Reviews may contain accented characters (cafe, resume), fancy quotes, ligatures (fi, fl), or other unicode variants. We normalize to NFC form
#first, then use ASCII transliteration to map accented chars to base chars. This ensures "cafe" and "cafe" hash identically.

def normalize_unicode(text: str) -> str:        #Normalizes unicode characters in review text.
    """
    Steps:
      1. NFC normalization composes precomposed characters (standard form).
      2. NFKD decomposition separates base characters from diacritics.
      3. We keep only ASCII-encodable characters (drops pure diacritics).
    """
    if not text:
        return ""

    #NFC first: compose any decomposed characters into their canonical form
    text = unicodedata.normalize("NFC", text)

    #NFKD decomposition then ASCII encoding strips diacritics while keeping
    #the base character. "ignore" drops characters that cannot be encoded.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    return text


#EMOJI HANDLING
#Emojis are meaningful signals in reviews ("Love it!") and should not be discarded. We convert emoji characters to their text description so they
#become shingle features. For example: "" becomes ":fire:" which produces shingles {":fi", "fir", "ire", "re:"}

def normalize_emoji(text: str) -> str:          #Converts emoji characters to their text description tokens.
    if not text:
        return ""

    result = []
    for char in text:
        #Check if this character is in a known emoji unicode range.
        #The ranges below cover the vast majority of commonly used emoji.
        cp = ord(char)
        if (
            0x1F600 <= cp <= 0x1F64F  #Emoticons
            or 0x1F300 <= cp <= 0x1F5FF  #Misc symbols and pictographs
            or 0x1F680 <= cp <= 0x1F6FF  #Transport and map symbols
            or 0x1F700 <= cp <= 0x1F77F  #Alchemical symbols
            or 0x2600 <= cp <= 0x26FF    #Misc symbols
            or 0x2700 <= cp <= 0x27BF    #Dingbats
            or 0xFE00 <= cp <= 0xFE0F    #Variation selectors
            or 0x1F900 <= cp <= 0x1F9FF  #Supplemental symbols
            or 0x1FA00 <= cp <= 0x1FA6F  #Chess symbols
            or 0x1FA70 <= cp <= 0x1FAFF  #Symbols and pictographs extended
        ):
            #Get the unicode name and convert to a token like :fire:
            try:
                name = unicodedata.name(char, "").lower().replace(" ", "_")
                if name:
                    result.append(f" :{name}: ")
                else:
                    result.append(" ")
            except (ValueError, TypeError):
                result.append(" ")
        else:
            result.append(char)

    return "".join(result)


#NON-ENGLISH DETECTION
#Some reviews in the All_Beauty dataset are written in non-English languages even though the category is English-language. Per the project spec, we
#flag these but do NOT discard them, since their shingles are still valid features for the similarity search.

def detect_language(text: str) -> str:          #Detects the language of a review using character distribution analysis. Returns an ISO 639-1 language code (e.g. "en", "es", "fr").
                                                #Returns "unknown" if detection fails or text is too short for reliable detection.
    if not text or len(text.strip()) < 20:
        #langdetect needs at least ~20 characters for reliable detection
        return "unknown"

    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


#TEXT CLEANING
#After HTML stripping and unicode normalization, we lowercase the text, remove punctuation, collapse whitespace, and remove stopwords. This produces the clean token list that feeds into the shingler.

#Build a translation table that maps every punctuation character to a space.
#Using str.translate with a prebuilt table is much faster than regex for this.
_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))


def clean_text(text: str) -> str:           #Lowercases text, removes punctuation, collapses whitespace, and removes English stopwords.
    
    if not text:
        return ""

    #Lowercase everything so "Amazing" and "amazing" produce the same shingles
    text = text.lower()

    #Replace punctuation characters with spaces using the prebuilt table
    text = text.translate(_PUNCT_TABLE)

    #Collapse multiple consecutive spaces into a single space
    text = re.sub(r"\s+", " ", text).strip()

    #Remove stopwords: split into words, filter, then rejoin.
    #We keep words that are NOT in the stopword set AND are non-empty.
    words = [w for w in text.split() if w not in ENGLISH_STOPWORDS and w]

    return " ".join(words)


#SHINGLE TOKENIZER
#This is the core data transformation that feeds MinHash.

#A character n-gram (shingle) of size k is every contiguous substring of length k in the text. We work on the full cleaned string (not word by word)
#so shingles can span word boundaries, which makes them more robust to minor wording variations.

#Example with SHINGLE_SIZE=3 on "great product":
#"gre", "rea", "eat", "at ", "t p", " pr", "pro", "rod", "odu", "duc", "uct"

#We return a SET of shingles (not a list) because:
#1. MinHash and Jaccard similarity are defined on sets
#2. Sets automatically deduplicate repeated shingles

def get_shingles(text: str, shingle_size: int = SHINGLE_SIZE) -> set:        #Tokenizes cleaned text into a set of overlapping character n-grams.
    if not text or len(text) < shingle_size:
        return set()

    #Slide a window of size shingle_size across the entire string.
    #range(len(text) - shingle_size + 1) gives us all valid start positions.
    shingles = set()
    for i in range(len(text) - shingle_size + 1):
        shingles.add(text[i : i + shingle_size])

    return shingles


#MASTER PREPROCESS FUNCTION
#This is the single entry point that all other modules call. It runs the full pipeline in order and returns a structured result dict.

def preprocess(text: str, shingle_size: int = SHINGLE_SIZE) -> dict:        #Runs the full preprocessing pipeline on a raw review text string.
    """
    Pipeline order:
      1. Input validation (handle None, non-string, empty)
      2. HTML stripping
      3. Emoji normalization (before unicode normalization eats them)
      4. Unicode normalization
      5. Language detection (on the cleaned but not yet lowercased text)
      6. Text cleaning (lowercase, remove punctuation, remove stopwords)
      7. Shingle tokenization
    """

    flags = []

    #Step 1: Input validation
    #Handle None, non-string types, and empty input gracefully.
    #Reviews that fail here get the "empty_input" flag and return early.
    if text is None or not isinstance(text, str) or text.strip() == "":
        return {
            "cleaned_text": "",
            "shingles": set(),
            "shingle_count": 0,
            "language": "unknown",
            "is_english": False,
            "is_usable": False,
            "flags": ["empty_input"],
        }

    raw_text = text.strip()

    #Step 2: HTML stripping
    #Detect HTML before stripping so we can set the flag
    if re.search(r"<[^>]+>|&[a-z]+;|&#\d+;", raw_text, re.IGNORECASE):
        flags.append("html_detected")

    html_stripped = strip_html(raw_text)

    #Step 3: Emoji normalization
    #Must happen before unicode normalization since NFKD encoding would
    #destroy emoji code points. We detect emoji by checking if any
    #characters were changed by normalize_emoji.
    emoji_normalized = normalize_emoji(html_stripped)
    if emoji_normalized != html_stripped:
        flags.append("emoji_detected")

    #Step 4: Unicode normalization
    unicode_normalized = normalize_unicode(emoji_normalized)

    #Step 5: Language detection
    #Run on the normalized but not yet lowercased text for best accuracy.
    language = detect_language(unicode_normalized)
    is_english = language == "en"
    if not is_english and language != "unknown":
        flags.append("non_english")

    #Step 6: Text cleaning
    cleaned = clean_text(unicode_normalized)

    #Step 7: Shingle tokenization
    shingles = get_shingles(cleaned, shingle_size=shingle_size)
    shingle_count = len(shingles)

    #Flag reviews that are too short to be meaningful after cleaning
    is_usable = shingle_count >= MIN_SHINGLES
    if not is_usable:
        flags.append("too_short")

    return {
        "cleaned_text": cleaned,
        "shingles": shingles,
        "shingle_count": shingle_count,
        "language": language,
        "is_english": is_english,
        "is_usable": is_usable,
        "flags": flags,
    }


#VALIDATION: SPOT CHECK ON 10 SAMPLE REVIEWS
#Running this file directly validates the preprocessor on real dataset records.
#This satisfies the Phase 1 requirement to spot-check shingle sets for 10 sample reviews.

if __name__ == "__main__":
    import sys
    import os

    #Add project root to path so we can import data_loader
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_records

    print("=" * 65)
    print("  NLPrint: Preprocessor Validation")
    print("  Spot-checking shingle output on 10 sample reviews")
    print("=" * 65)

    sample_count = 0
    usable_count = 0
    flagged_count = 0

    for record in load_records():
        if sample_count >= 10:
            break

        raw_text = record.get("text", "")
        result = preprocess(raw_text)

        sample_count += 1
        if result["is_usable"]:
            usable_count += 1
        if result["flags"]:
            flagged_count += 1

        print(f"\n--- Sample {sample_count} ---")
        print(f"  Raw text (first 80 chars): {raw_text[:80]!r}")
        print(f"  Cleaned text:              {result['cleaned_text'][:80]!r}")
        print(f"  Language:                  {result['language']}")
        print(f"  Is English:                {result['is_english']}")
        print(f"  Shingle count:             {result['shingle_count']}")
        print(f"  Is usable:                 {result['is_usable']}")
        print(f"  Flags:                     {result['flags']}")

        #Show a sample of 5 shingles so we can visually verify them
        sample_shingles = sorted(result["shingles"])[:5]
        print(f"  First 5 shingles:          {sample_shingles}")

    print("\n" + "=" * 65)
    print(f"  Samples checked:  {sample_count}")
    print(f"  Usable reviews:   {usable_count}")
    print(f"  Flagged reviews:  {flagged_count}")
    print("=" * 65)
    print("  Preprocessor validation complete.")
    print("=" * 65)