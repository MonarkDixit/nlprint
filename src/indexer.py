#src/indexer.py
#NLPrint: Amazon Review Authenticity Detector
#Author: Monark Dixit (UID: 122259645)
#Course: MSML606, Spring 2026

#PURPOSE:
#This module builds and manages the in-memory signature index for NLPrint.

#The index is the core data structure that makes real-time similarity search possible. It stores a MinHash signature vector for every review
#in the dataset, along with the review's metadata. At query time, the query engine scans this index to find the most similar reviews.

#INDEX STRUCTURE:
#The index is a list of IndexEntry objects, one per review. Each entry holds:
#  - The MinHash signature vector (list of k integers)
#  - The review metadata (text, rating, user_id, timestamp, etc.)
#  - The preprocessed result (shingle count, flags, language)

#BUILD COMPLEXITY: O(n * k)
#n = number of reviews (up to 10,000)
#k = number of hash functions (default 100)
#This is far better than O(n^2) brute-force pairwise comparison.

#The index can be saved to disk as a JSON file so it does not need to be rebuilt every time the app starts. On subsequent runs, it is loaded from disk in seconds.
import json
import os
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import preprocess, SHINGLE_SIZE
from src.minhash import MinHasher, DEFAULT_K
from src.hasher import FeatureHasher, DEFAULT_TABLE_SIZE
from src.data_loader import load_records, download_dataset


#CONSTANTS
#Path where the built index is saved so we don't rebuild on every run
INDEX_SAVE_PATH = os.path.join("data", "signature_index.json")


#INDEX ENTRY
#A lightweight container for one review's data in the index. We use a plain dict internally for easy JSON serialization.

def make_index_entry(
    record: dict,
    signature: list,
    shingle_count: int,
    language: str,
    flags: list,
    load_factor: float,
) -> dict:          #Creates a single index entry dict from a record and its computed values.
    return {
        #Core hashing data
        "signature": signature,             #MinHash signature (list of k ints)
        "shingle_count": shingle_count,     #Used for stats display
        "load_factor": load_factor,         #Used for stats sidebar

        #Review content for the results panel
        "text": str(record.get("text", "") or ""),
        "title": str(record.get("title", "") or ""),
        "rating": record.get("rating", None),

        #Metadata fields used for suspicious signal detection
        "user_id": str(record.get("user_id", "") or ""),
        "timestamp": record.get("timestamp", None),
        "verified_purchase": record.get("verified_purchase", None),
        "helpful_vote": record.get("helpful_vote", 0),

        #Preprocessing metadata
        "language": language,
        "flags": flags,
    }


#SIGNATURE INDEX
#The main index class. Builds, stores, and manages all review signatures.

class SignatureIndex:           #Manages the in-memory MinHash signature index for all reviews.

    def __init__(
        self,
        k: int = DEFAULT_K,
        shingle_size: int = SHINGLE_SIZE,
        table_size: int = DEFAULT_TABLE_SIZE,
        seed: int = 42,
    ):
        
        #Initializes the index with the given algorithm parameters.

        self.k = k
        self.shingle_size = shingle_size
        self.table_size = table_size
        self.seed = seed

        #Initialize the MinHasher and FeatureHasher with matching parameters
        self.minhash = MinHasher(k=k, seed=seed)
        self.hasher = FeatureHasher(table_size=table_size, seed=seed)

        #The entries list holds one dict per indexed review
        self.entries = []

        #Timing data for the stats sidebar
        self.build_time_seconds = 0.0

    def build(self, max_records: int = None, verbose: bool = True) -> int:          #Builds the index by iterating all records in the dataset.
        
        #Make sure the dataset is downloaded first
        download_dataset()

        self.entries = []
        skipped = 0
        indexed = 0
        start_time = time.time()

        if verbose:
            print(f"[INFO] Building signature index...")
            print(f"[INFO] Parameters: k={self.k}, shingle_size={self.shingle_size}, "
                  f"table_size={self.table_size}")
            
            record_display = "all" if max_records is None else f"{max_records:,}"
            print(f"[INFO] This will process up to {record_display} records.\n")

        for record in load_records(skip_empty_text=True):
            if max_records is not None and indexed + skipped >= max_records:
                break

            #Run the full preprocessing pipeline on this review's text
            raw_text = record.get("text", "")
            prep = preprocess(raw_text, shingle_size=self.shingle_size)

            #Skip reviews that are too short after cleaning to be meaningful
            if not prep["is_usable"]:
                skipped += 1
                continue

            #Compute the MinHash signature for this review's shingle set
            signature = self.minhash.get_signature(prep["shingles"])

            #Compute the feature hash load factor for stats display
            load_factor = self.hasher.get_load_factor(prep["shingles"])

            #Build and store the index entry
            entry = make_index_entry(
                record=record,
                signature=signature,
                shingle_count=prep["shingle_count"],
                language=prep["language"],
                flags=prep["flags"],
                load_factor=load_factor,
            )
            self.entries.append(entry)
            indexed += 1

            #Print progress every 1000 records so the user knows it is running
            if verbose and indexed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = indexed / elapsed if elapsed > 0 else 0
                print(f"  Indexed {indexed:,} records "
                      f"({rate:.0f} records/sec, {elapsed:.1f}s elapsed)")

        self.build_time_seconds = time.time() - start_time

        if verbose:
            print(f"\n[INFO] Index build complete.")
            print(f"  Records indexed:  {indexed:,}")
            print(f"  Records skipped:  {skipped:,}")
            print(f"  Build time:       {self.build_time_seconds:.2f}s")
            rate = indexed / self.build_time_seconds if self.build_time_seconds > 0 else 0
            print(f"  Throughput:       {rate:.0f} records/sec")

        return indexed

    def save(self, path: str = INDEX_SAVE_PATH) -> None:        #Saves the index to a JSON file on disk.
    
        os.makedirs(os.path.dirname(path), exist_ok=True)

        #Bundle entries with metadata about how the index was built
        payload = {
            "meta": {
                "k": self.k,
                "shingle_size": self.shingle_size,
                "table_size": self.table_size,
                "seed": self.seed,
                "record_count": len(self.entries),
                "build_time_seconds": self.build_time_seconds,
            },
            "entries": self.entries,
        }

        print(f"[INFO] Saving index ({len(self.entries):,} entries) to '{path}'...")
        save_start = time.time()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        save_elapsed = time.time() - save_start
        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[INFO] Saved in {save_elapsed:.2f}s. File size: {file_size_mb:.1f} MB")

    def load(self, path: str = INDEX_SAVE_PATH) -> bool:        #Loads a previously saved index from disk.
        if not os.path.exists(path):
            print(f"[INFO] No saved index found at '{path}'.")
            return False

        print(f"[INFO] Loading index from '{path}'...")
        load_start = time.time()

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        meta = payload.get("meta", {})

        #Warn if the saved index used different parameters than the current instance. Different k or shingle_size means the signatures are incompatible with new queries.
        if meta.get("k") != self.k or meta.get("shingle_size") != self.shingle_size:
            print(f"[WARNING] Saved index parameters differ from current settings.")
            print(f"  Saved:   k={meta.get('k')}, shingle_size={meta.get('shingle_size')}")
            print(f"  Current: k={self.k}, shingle_size={self.shingle_size}")
            print(f"  Delete '{path}' and rebuild the index for correct results.")

        self.entries = payload.get("entries", [])

        #Restore build metadata
        self.build_time_seconds = meta.get("build_time_seconds", 0.0)

        load_elapsed = time.time() - load_start
        print(f"[INFO] Loaded {len(self.entries):,} entries in {load_elapsed:.2f}s.")
        return True

    def load_or_build(                  #Convenience method: loads the index from disk if it exists, otherwise builds it and saves it.
        self,
        path: str = INDEX_SAVE_PATH,
        max_records: int = None,
        verbose: bool = True,
    ) -> int:
        if self.load(path):
            return len(self.entries)

        print("[INFO] Building index from scratch...")
        count = self.build(max_records=max_records, verbose=verbose)
        self.save(path)
        return count

    def get_stats(self) -> dict:            #Returns summary statistics about the current index state.
        if not self.entries:
            return {
                "entry_count": 0,
                "avg_load_factor": 0.0,
                "avg_shingle_count": 0.0,
                "build_time_seconds": self.build_time_seconds,
                "k": self.k,
                "shingle_size": self.shingle_size,
                "table_size": self.table_size,
            }

        avg_load = sum(e["load_factor"] for e in self.entries) / len(self.entries)
        avg_shingles = sum(e["shingle_count"] for e in self.entries) / len(self.entries)

        return {
            "entry_count": len(self.entries),
            "avg_load_factor": round(avg_load, 4),
            "avg_shingle_count": round(avg_shingles, 1),
            "build_time_seconds": round(self.build_time_seconds, 2),
            "k": self.k,
            "shingle_size": self.shingle_size,
            "table_size": self.table_size,
        }

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return (
            f"SignatureIndex("
            f"entries={len(self.entries)}, "
            f"k={self.k}, "
            f"shingle_size={self.shingle_size})"
        )


#ENTRY POINT: BUILD AND BENCHMARK
#Running this file directly builds the full index and reports timing. This satisfies the Phase 2 benchmark requirement.

if __name__ == "__main__":
    print("=" * 65)
    print("  NLPrint: Index Builder Benchmark")
    print("=" * 65)

    index = SignatureIndex(k=DEFAULT_K, shingle_size=SHINGLE_SIZE)
    count = index.load_or_build(verbose=True)

    print(f"\n  Index stats:")
    stats = index.get_stats()
    for key, val in stats.items():
        print(f"    {key:<28} {val}")

    print("\n" + "=" * 65)
    print("  Index build benchmark complete.")
    print("=" * 65)