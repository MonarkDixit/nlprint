#src/query.py
#NLPrint: Amazon Review Authenticity Detector
#Author: Monark Dixit (UID: 122259645)
#Course: MSML606, Spring 2026

#PURPOSE:
#This module implements the query and lookup engine for NLPrint.

#Given any raw review text as input, the query engine:
#  1. Runs the text through the full preprocessing pipeline
#  2. Computes the MinHash signature of the input
#  3. Scans the signature index to find the top-N most similar reviews
#  4. Computes metadata flags for suspicious signals:
#       - Shared user_id between the input and a matched review
#       - Timestamp proximity (reviews submitted very close in time)
#       - verified_purchase mismatches
#  5. Returns ranked results ready for the Streamlit UI

#COMPLEXITY:
#Query time is O(n * k) where n is the index size and k is the number of hash functions. For n=10,000 and k=100, this is 1,000,000 simple integer comparisons, 
#which runs in well under 1 second on modern hardware. This is the key advantage over brute-force O(n^2) comparison.

import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import preprocess, SHINGLE_SIZE
from src.minhash import MinHasher, DEFAULT_K
from src.indexer import SignatureIndex, INDEX_SAVE_PATH


#CONSTANTS
#Default number of top results to return
DEFAULT_TOP_N = 5

#Timestamp proximity threshold in seconds.
#Reviews submitted within this window of each other are flagged as suspiciously close in time. 86400 seconds = 24 hours.
TIMESTAMP_PROXIMITY_THRESHOLD = 86400  #24 hours in seconds

#Minimum Jaccard similarity to include a result in the output. Results below this threshold are too dissimilar to be interesting.
MIN_SIMILARITY_THRESHOLD = 0.05


#METADATA FLAG DETECTION
#These helper functions analyze pairs of reviews for suspicious signals.
#The results are displayed in the UI metadata flags panel.

def check_shared_user_id(input_user_id: str, match_user_id: str) -> bool:
    #Returns True if the input review and the matched review share the same user_id. A shared user_id means both reviews were written by the same
    #account, which is a strong signal of copy-pasted content or review fraud.
    if not input_user_id or not match_user_id:
        return False
    return input_user_id.strip() == match_user_id.strip()


def check_timestamp_proximity(ts_a, ts_b) -> bool:
    
    #Returns True if two review timestamps are within TIMESTAMP_PROXIMITY_THRESHOLD seconds of each other.
    if ts_a is None or ts_b is None:
        return False
    try:
        #Timestamps in this dataset are in milliseconds, convert to seconds
        a = int(ts_a) / 1000
        b = int(ts_b) / 1000
        return abs(a - b) <= TIMESTAMP_PROXIMITY_THRESHOLD
    except (ValueError, TypeError):
        return False


def check_verified_mismatch(vp_input, vp_match) -> bool:
    
    #Returns True if the verified_purchase status differs between the input review and the matched review. A high-similarity review pair where one
    #is verified and the other is not may indicate a copied unverified review.
    if vp_input is None or vp_match is None:
        return False
    return bool(vp_input) != bool(vp_match)


def format_timestamp(ts) -> str:
    
    #Converts a Unix epoch timestamp (in milliseconds) to a human-readable date string for display in the UI.
    if ts is None:
        return "Unknown"
    try:
        dt = datetime.utcfromtimestamp(int(ts) / 1000)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return "Unknown"


#QUERY RESULT
#A single result from a query. Each result represents one indexed review that is similar to the input, along with its similarity score and flags.

def make_query_result(entry: dict, similarity: float, input_meta: dict) -> dict:
    
    #Builds a single query result dict from an index entry and its similarity score. Computes all metadata flags for the results panel.
    #Compute metadata flags by comparing the input review to this match
    shared_user = check_shared_user_id(
        input_meta.get("user_id", ""),
        entry.get("user_id", ""),
    )
    ts_proximity = check_timestamp_proximity(
        input_meta.get("timestamp"),
        entry.get("timestamp"),
    )
    vp_mismatch = check_verified_mismatch(
        input_meta.get("verified_purchase"),
        entry.get("verified_purchase"),
    )

    #Build the list of active flags for this result
    metadata_flags = []
    if shared_user:
        metadata_flags.append("shared_user_id")
    if ts_proximity:
        metadata_flags.append("timestamp_proximity")
    if vp_mismatch:
        metadata_flags.append("verified_purchase_mismatch")
    if entry.get("language", "en") != "en":
        metadata_flags.append(f"non_english ({entry.get('language', '?')})")

    return {
        #Similarity score for the badge display
        "similarity": round(similarity, 4),
        "similarity_pct": f"{similarity * 100:.1f}%",

        #Review content for the results panel
        "text": entry.get("text", ""),
        "title": entry.get("title", ""),
        "rating": entry.get("rating", None),
        "helpful_vote": entry.get("helpful_vote", 0),

        #Metadata for the flags panel
        "user_id": entry.get("user_id", ""),
        "timestamp": entry.get("timestamp", None),
        "timestamp_display": format_timestamp(entry.get("timestamp")),
        "verified_purchase": entry.get("verified_purchase", None),

        #Flags
        "metadata_flags": metadata_flags,
        "has_flags": len(metadata_flags) > 0,

        #Preprocessing metadata
        "language": entry.get("language", "unknown"),
        "shingle_count": entry.get("shingle_count", 0),
        "load_factor": entry.get("load_factor", 0.0),
    }


#QUERY ENGINE
#The main class that ties preprocessing, MinHash, and the index together.

class QueryEngine:
    
    #Accepts a raw review text input and returns the top-N most similar reviews from the signature index, with similarity scores and metadata flags.


    def __init__(self, index: SignatureIndex):      #Initializes the query engine with a built signature index.

        self.index = index

        #The query engine's MinHasher must use identical parameters to the one that built the index, or signatures will be incompatible.
        self.minhash = MinHasher(k=index.k, seed=index.seed)

    def query(
        self,
        raw_text: str,
        top_n: int = DEFAULT_TOP_N,
        input_user_id: str = "",
        input_timestamp=None,
        input_verified_purchase=None,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ) -> dict:              #Queries the index for reviews most similar to the input text.
        query_start = time.time()

        #Input validation
        if not raw_text or not isinstance(raw_text, str) or raw_text.strip() == "":
            return {
                "results": [],
                "query_text": raw_text,
                "cleaned_text": "",
                "shingle_count": 0,
                "language": "unknown",
                "flags": ["empty_input"],
                "is_usable": False,
                "query_time_ms": 0.0,
                "index_size": len(self.index),
                "error": "Input text is empty. Please enter a review to check.",
            }

        if len(self.index) == 0:
            return {
                "results": [],
                "query_text": raw_text,
                "cleaned_text": "",
                "shingle_count": 0,
                "language": "unknown",
                "flags": [],
                "is_usable": False,
                "query_time_ms": 0.0,
                "index_size": 0,
                "error": "Index is empty. Please build the index first.",
            }

        #Step 1: Preprocess the input
        prep = preprocess(raw_text, shingle_size=self.index.shingle_size)

        if not prep["is_usable"]:
            return {
                "results": [],
                "query_text": raw_text,
                "cleaned_text": prep["cleaned_text"],
                "shingle_count": prep["shingle_count"],
                "language": prep["language"],
                "flags": prep["flags"],
                "is_usable": False,
                "query_time_ms": (time.time() - query_start) * 1000,
                "index_size": len(self.index),
                "error": (
                    "Input review is too short after cleaning. "
                    "Please enter a longer review."
                ),
            }

        #Step 2: Compute MinHash signature for the input
        query_signature = self.minhash.get_signature(prep["shingles"])

        #Build the input metadata dict for flag comparison
        input_meta = {
            "user_id": input_user_id,
            "timestamp": input_timestamp,
            "verified_purchase": input_verified_purchase,
        }

        #Step 3: Scan the index
        #For each indexed entry, estimate Jaccard similarity by comparing the query signature against the stored signature.
        #This is O(n * k) integer comparisons.
        scored = []
        for entry in self.index.entries:
            stored_sig = entry["signature"]
            similarity = self.minhash.estimate_jaccard(query_signature, stored_sig)

            if similarity >= min_similarity:
                result = make_query_result(entry, similarity, input_meta)
                scored.append(result)

        #Step 4: Sort by similarity descending and take top-N
        scored.sort(key=lambda r: r["similarity"], reverse=True)
        top_results = scored[:top_n]

        query_time_ms = (time.time() - query_start) * 1000

        return {
            "results": top_results,
            "query_text": raw_text,
            "cleaned_text": prep["cleaned_text"],
            "shingle_count": prep["shingle_count"],
            "language": prep["language"],
            "flags": prep["flags"],
            "is_usable": True,
            "query_time_ms": round(query_time_ms, 2),
            "index_size": len(self.index),
            "error": None,
        }


#ENTRY POINT: BENCHMARK QUERY LATENCY
#Running this file builds (or loads) the index and benchmarks query speed. This satisfies the Phase 2 benchmark requirement for query latency.

if __name__ == "__main__":
    print("=" * 65)
    print("  NLPrint: Query Engine Benchmark")
    print("=" * 65)

    #Build or load the index
    print("\n[STEP 1] Loading or building index...")
    index = SignatureIndex(k=DEFAULT_K, shingle_size=SHINGLE_SIZE)
    index.load_or_build(verbose=True)

    engine = QueryEngine(index)

    #Test queries
    test_queries = [
        "This moisturizer is absolutely amazing. My skin feels so soft and hydrated.",
        "Terrible product. It broke after one use and smells awful. Waste of money.",
        "Great value for the price. Works exactly as described. Would buy again.",
        "I love this product! It smells wonderful and makes my hair so shiny.",
    ]

    print(f"\n[STEP 2] Running {len(test_queries)} test queries...\n")

    for i, query_text in enumerate(test_queries, 1):
        response = engine.query(query_text, top_n=DEFAULT_TOP_N)

        print(f"  Query {i}: {query_text[:60]!r}")
        print(f"    Shingle count:    {response['shingle_count']}")
        print(f"    Language:         {response['language']}")
        print(f"    Results found:    {len(response['results'])}")
        print(f"    Query time:       {response['query_time_ms']:.1f}ms")
        print(f"    Index size:       {response['index_size']:,}")

        if response["results"]:
            top = response["results"][0]
            print(f"    Top match ({top['similarity_pct']}): {top['text'][:60]!r}")
            if top["metadata_flags"]:
                print(f"    Flags:            {top['metadata_flags']}")
        print()

    #Latency benchmark: run 50 queries and average
    print("  [BENCHMARK] Average query latency over 50 runs")
    benchmark_text = "This product is really great and I love it so much."
    times = []
    for _ in range(50):
        r = engine.query(benchmark_text)
        times.append(r["query_time_ms"])

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    print(f"    Average latency: {avg_ms:.1f}ms")
    print(f"    Min latency:     {min_ms:.1f}ms")
    print(f"    Max latency:     {max_ms:.1f}ms")

    print("\n" + "=" * 65)
    print("  Query engine benchmark complete.")
    print("=" * 65)