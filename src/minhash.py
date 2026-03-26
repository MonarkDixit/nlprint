#src/minhash.py
#NLPrint: Amazon Review Authenticity Detector
#Author: Monark Dixit (UID: 122259645)
#Course: MSML606, Spring 2026

#PURPOSE:
#This module implements MinHash signatures from scratch.

#WHAT IS MINHASH?
#MinHash is a locality-sensitive hashing (LSH) technique that produces a compact "signature" for a set such that the similarity between two signatures approximates the Jaccard similarity of the original sets.

#THE CORE MATHEMATICAL PROPERTY:
#Given two sets A and B, and a random hash function h:
#P(min(h(A)) == min(h(B))) = |A intersection B| / |A union B| = Jaccard(A, B)


#WHY USE k HASH FUNCTIONS?
#One hash function gives us a probability estimate, but it is noisy. By using k independent hash functions and taking the minimum under each,
#we get a signature vector of length k. The estimated Jaccard similarity between two reviews is then:
#Jaccard_estimate = (number of positions where sigs agree) / k

#The variance of this estimate decreases as k increases: std_error = sqrt(J(1-J) / k)
#With k=100, the standard error is about 0.05 at J=0.5. With k=200, it drops to about 0.035. We use k=100 as a good speed/accuracy balance.

#COMPLEXITY IMPROVEMENT:
#Brute force pairwise comparison of n reviews: O(n^2 * shingle_size)
#MinHash index build:                          O(n * k)
#MinHash query for one review:                 O(n * k)
#This makes real-time querying viable at 10,000+ records.

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hasher import UniversalHashFunction, MERSENNE_PRIME


#CONSTANTS
#Number of hash functions in the MinHash signature. Higher k = more accurate Jaccard estimates but more memory and compute. k=100 gives std_error ~ 0.05 at J=0.5, which is good for our use case.
DEFAULT_K = 100

#We use a large integer as the "infinity" sentinel when computing minimums. Any real hash value will be smaller than this.
INF = float("inf")


#MINHASH SIGNATURE GENERATOR
#Applies k independent universal hash functions to a shingle set and returns the vector of k minimum hash values as the signature.

class MinHasher:        #Generates MinHash signatures for shingle sets.

    def __init__(self, k: int = DEFAULT_K, seed: int = 42):         #Initializes k independent universal hash functions.
        self.k = k
        #We use a large table_size here because for MinHash we need the raw integer output of the hash (to find the minimum), not a bucket index. 
        #Using MERSENNE_PRIME as table_size effectively gives us the full range of the universal hash function.
        
        #Each function gets seed + i so they are independent of each other but reproducible given the same base seed.
        self.hash_functions = [
            UniversalHashFunction(table_size=MERSENNE_PRIME, seed=seed + i)
            for i in range(k)
        ]

    def get_signature(self, shingles: set) -> list:         #Computes the MinHash signature for a shingle set.
        if not shingles:
            #Empty set has no similarity with anything
            return [MERSENNE_PRIME] * self.k

        #Pre-convert all shingles to their integer hash keys once. This avoids re-hashing the same string k times. 
        #We use abs(hash(s)) & 0x7FFFFFFF as the pre-hash step, consistent with UniversalHashFunction.hash_string().
        FNV_PRIME = 16777619
        FNV_OFFSET = 2166136261

        shingle_ints = []
        for s in shingles:
            h = FNV_OFFSET
            for byte in s.encode("utf-8"):
                h ^= byte
                h = (h * FNV_PRIME) & 0xFFFFFFFF
            shingle_ints.append(h)

        #For each hash function, find the minimum hash value over all shingles. This is the core MinHash computation.
        signature = []
        for h in self.hash_functions:
            min_val = MERSENNE_PRIME  #Start with the maximum possible value

            for x in shingle_ints:
                #Apply the universal hash function to this shingle's integer
                hashed = h.hash_int(x)
                if hashed < min_val:
                    min_val = hashed

            signature.append(min_val)

        return signature

    def estimate_jaccard(self, sig_a: list, sig_b: list) -> float:      #Estimates Jaccard similarity between two sets from their signatures.
        if not sig_a or not sig_b or len(sig_a) != len(sig_b):
            return 0.0

        #Count positions where both signatures agree (same minimum value)
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)

        return matches / self.k

    def __repr__(self):
        return f"MinHasher(k={self.k}, hash_functions={len(self.hash_functions)})"


#ENTRY POINT: BENCHMARK AND VALIDATION
#Running this file directly validates MinHash on known pairs and reports accuracy of the Jaccard estimate vs the true Jaccard similarity.

if __name__ == "__main__":
    import time
    from src.preprocessor import preprocess

    print("=" * 65)
    print("  NLPrint: MinHash Validation and Benchmark")
    print("=" * 65)

    minhash = MinHasher(k=DEFAULT_K, seed=42)

    #Test pairs with known expected Jaccard similarity
    test_pairs = [
        (
            #Pair 1: Exact duplicates - expected Jaccard = 1.0
            "This product is absolutely amazing and works perfectly well.",
            "This product is absolutely amazing and works perfectly well.",
            "Exact duplicate",
        ),
        (
            #Pair 2: Near duplicates - expected Jaccard ~0.7-0.9
            "This product is absolutely amazing and works perfectly well.",
            "This product is absolutely amazing and works really well for me.",
            "Near duplicate",
        ),
        (
            #Pair 3: Somewhat similar - expected Jaccard ~0.3-0.5
            "Great product, smells wonderful and feels nice on skin.",
            "Good product overall, nice scent and smooth texture.",
            "Somewhat similar",
        ),
        (
            #Pair 4: Completely different - expected Jaccard ~0.0-0.1
            "This moisturizer is the best I have ever used in my life.",
            "Terrible quality, broke after one day, total waste of money.",
            "Completely different",
        ),
    ]

    print(f"\n  MinHash parameters: k={DEFAULT_K}")
    print(f"  Shingle size: 3 (trigrams)\n")

    for text_a, text_b, label in test_pairs:
        #Preprocess both texts
        result_a = preprocess(text_a)
        result_b = preprocess(text_b)

        #Compute true Jaccard similarity for comparison
        set_a = result_a["shingles"]
        set_b = result_b["shingles"]

        if set_a or set_b:
            true_jaccard = len(set_a & set_b) / len(set_a | set_b)
        else:
            true_jaccard = 0.0

        #Compute MinHash signatures
        sig_a = minhash.get_signature(set_a)
        sig_b = minhash.get_signature(set_b)

        #Estimate Jaccard from signatures
        estimated = minhash.estimate_jaccard(sig_a, sig_b)
        error = abs(true_jaccard - estimated)

        print(f"  [{label}]")
        print(f"    True Jaccard:      {true_jaccard:.4f}")
        print(f"    MinHash estimate:  {estimated:.4f}")
        print(f"    Absolute error:    {error:.4f}")
        print()

    #Benchmark: how long does signing 10,000 reviews take?
    print("  [BENCHMARK] Signature generation speed")

    sample_text = "This product is amazing and I love using it every day."
    result = preprocess(sample_text)
    shingles = result["shingles"]

    n_reviews = 1000
    start = time.time()
    for _ in range(n_reviews):
        minhash.get_signature(shingles)
    elapsed = time.time() - start

    per_review = elapsed / n_reviews * 1000
    projected_10k = elapsed / n_reviews * 10000

    print(f"    Reviews signed:          {n_reviews:,}")
    print(f"    Total time:              {elapsed:.3f}s")
    print(f"    Per review:              {per_review:.3f}ms")
    print(f"    Projected for 10k:       {projected_10k:.1f}s")

    print("\n" + "=" * 65)
    print("  MinHash validation complete.")
    print("=" * 65)