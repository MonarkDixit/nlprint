#src/hasher.py
#NLPrint: Amazon Review Authenticity Detector
#Author: Monark Dixit (UID: 122259645)
#Course: MSML606, Spring 2026

#PURPOSE:
#This module implements two core data structures from scratch:

#1. UNIVERSAL HASH FUNCTION
#A universal hash function maps an arbitrary input to a fixed integer range [0, table_size). The "universal" property means that for any
#two distinct keys x and y, the probability that h(x) == h(y) is at most 1/table_size, regardless of the input distribution. This gives
#us strong collision resistance without relying on the keys being uniformly distributed.

#We implement the multiply-shift scheme:
#h(x) = ((a * x + b) mod p) mod table_size
#where:
#a, b  = random integers (chosen at construction time)
#p = a large prime number larger than table_size
#x = integer representation of the input key

#2. FEATURE HASHING (also called the "hashing trick")
#Feature hashing maps a set of shingle strings into a fixed-size integer array (the "hash table") by:
#  - Converting each shingle to an integer via a string hash
#  - Mapping that integer to a bucket index via the universal hash
#  - Incrementing the count at that bucket

#The result is a fixed-width vector representation of any shingle set, regardless of how many unique shingles the vocabulary contains. This is the core data structure NLPrint uses to represent reviews.

#MATH NOTE ON UNIVERSAL HASHING:
#The prime p must satisfy p > table_size and p > max possible key value. We use the Mersenne prime 2^31 - 1 = 2147483647, which is large enough for all practical shingle hash values.

#USAGE:
#from src.hasher import UniversalHashFunction, FeatureHasher


import random
import math


# CONSTANTS
#A large Mersenne prime used as the modulus in the universal hash function. Mersenne primes (of the form 2^n - 1) are efficient for modular arithmetic.
MERSENNE_PRIME = (1 << 31) - 1   # 2^31 - 1 = 2,147,483,647

#Default hash table size: number of buckets in the feature hash table.
#Powers of 2 are cache-friendly. 1024 gives a compact but informative representation. Increasing this reduces collisions but uses more memory.
DEFAULT_TABLE_SIZE = 1024


#UNIVERSAL HASH FUNCTION
#Implements the multiply-add-mod scheme from Carter and Wegman (1979). This is implemented entirely from scratch with no external hashing libraries.

class UniversalHashFunction:        
    #A single universal hash function of the form:
    #h(x) = ((a * x + b) mod p) mod table_size

    #Each instance has independently randomly chosen parameters a and b, so two instances of this class produce statistically independent hash
    #outputs for the same input. This independence property is what makes MinHash work correctly.
    
    def __init__(self, table_size: int = DEFAULT_TABLE_SIZE, seed: int = None):     #Initializes the hash function with randomly chosen a and b parameters.
    
        self.table_size = table_size
        self.p = MERSENNE_PRIME

        #Use a seeded RNG if provided, otherwise use system randomness.
        #Seeded RNG is used by MinHash so its k hash functions are reproducible across runs (same index, same query results).
        rng = random.Random(seed)

        #a must be in [1, p-1] to avoid the degenerate case h(x) = b mod m
        #for all x (which would map everything to the same bucket).
        self.a = rng.randint(1, self.p - 1)

        #b can be in [0, p-1]
        self.b = rng.randint(0, self.p - 1)

    def hash_int(self, x: int) -> int:      #Hashes an integer x to a bucket index in [0, table_size). Formula: h(x) = ((a * x + b) mod p) mod table_size
        return ((self.a * x + self.b) % self.p) % self.table_size

    def hash_string(self, s: str) -> int:       #Hashes a string to a bucket index in [0, table_size).
        #Convert string to a non-negative integer. We use the built-in hash as a pre-hash step, then apply our universal function on top. This gives us the universal collision
        #probability guarantee on top of Python's string hash distribution.
        int_val = abs(hash(s)) & 0x7FFFFFFF   #mask to 31 bits
        return self.hash_int(int_val)

    def __repr__(self):
        return (
            f"UniversalHashFunction("
            f"table_size={self.table_size}, a={self.a}, b={self.b})"
        )


#FEATURE HASHER
#Maps a shingle set (variable size) to a fixed-size integer array. This is the "hashing trick" used in large-scale machine learning and NLP.

class FeatureHasher:        #Maps a set of shingle strings to a fixed-size integer array (hash table) using a universal hash function.

    def __init__(self, table_size: int = DEFAULT_TABLE_SIZE, seed: int = 42):       #Initializes the feature hasher with a universal hash function.
        self.table_size = table_size
        # Create the hash function with a fixed seed so the feature representation is consistent across all records in the index.
        self.hash_fn = UniversalHashFunction(table_size=table_size, seed=seed)

    def hash_shingles(self, shingles: set) -> list:     #Maps a set of shingles to a fixed-size hash table (integer array).
        #Initialize all buckets to zero
        table = [0] * self.table_size

        if not shingles:
            return table

        for shingle in shingles:
            #Get the bucket index for this shingle
            bucket = self.hash_fn.hash_string(shingle)
            #Increment the count at that bucket
            table[bucket] += 1

        return table

    def get_load_factor(self, shingles: set) -> float:      #Computes the hash table load factor for a given shingle set.
        if not shingles:
            return 0.0

        table = self.hash_shingles(shingles)
        occupied = sum(1 for bucket in table if bucket > 0)
        return occupied / self.table_size

    def get_occupied_buckets(self, shingles: set) -> int:       #Returns the number of buckets that have at least one shingle.
        
        table = self.hash_shingles(shingles)
        return sum(1 for bucket in table if bucket > 0)

    def __repr__(self):
        return f"FeatureHasher(table_size={self.table_size})"


#ENTRY POINT: BENCHMARK AND VALIDATION
#Running this file directly tests the hash function and feature hasher on sample shingle sets and reports collision statistics.

if __name__ == "__main__":
    import sys
    import os
    import time

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocessor import preprocess

    print("=" * 65)
    print("  NLPrint: Hasher Validation and Benchmark")
    print("=" * 65)

    #Test 1: Universal hash function basic properties
    print("\n[TEST 1] Universal Hash Function Basic Properties")

    h1 = UniversalHashFunction(table_size=1024, seed=1)
    h2 = UniversalHashFunction(table_size=1024, seed=2)

    print(f"  h1 parameters: a={h1.a}, b={h1.b}")
    print(f"  h2 parameters: a={h2.a}, b={h2.b}")

    test_strings = ["hello", "world", "great", "product", "amazing", "hello"]
    print(f"\n  String hashing (h1 vs h2):")
    for s in test_strings:
        v1 = h1.hash_string(s)
        v2 = h2.hash_string(s)
        print(f"    '{s}' -> h1={v1:4d}, h2={v2:4d}")

    #Test 2: Collision rate check
    print("\n[TEST 2] Collision Rate Check (1000 random strings)")

    test_hash = UniversalHashFunction(table_size=1024, seed=99)
    buckets_hit = set()
    n_strings = 1000

    for i in range(n_strings):
        test_str = f"shingle_{i:04d}"
        bucket = test_hash.hash_string(test_str)
        buckets_hit.add(bucket)

    collision_rate = 1.0 - (len(buckets_hit) / n_strings)
    print(f"  Strings hashed:     {n_strings}")
    print(f"  Unique buckets hit: {len(buckets_hit)}")
    print(f"  Collision rate:     {collision_rate:.2%}")
    print(f"  Expected (1/m):     {1/1024:.2%}")

    #Test 3: Feature hasher on real reviews
    print("\n[TEST 3] Feature Hasher on Real Reviews")

    hasher = FeatureHasher(table_size=DEFAULT_TABLE_SIZE, seed=42)

    sample_reviews = [
        "This product is absolutely amazing and works perfectly.",
        "This product is absolutely amazing and works perfectly.",  #exact duplicate
        "Terrible product. Does not work at all. Very disappointed.",
        "Great smell, love the texture.",
    ]

    print(f"  Table size: {DEFAULT_TABLE_SIZE} buckets\n")

    tables = []
    for i, review in enumerate(sample_reviews):
        result = preprocess(review)
        table = hasher.hash_shingles(result["shingles"])
        load = hasher.get_load_factor(result["shingles"])
        occupied = hasher.get_occupied_buckets(result["shingles"])
        tables.append(table)

        label = "(DUPLICATE)" if i == 1 else ""
        print(f"  Review {i+1} {label}")
        print(f"    Text (40 chars): {review[:40]!r}")
        print(f"    Shingle count:   {result['shingle_count']}")
        print(f"    Occupied buckets:{occupied}")
        print(f"    Load factor:     {load:.3f}")

    #Verify exact duplicates produce identical tables
    print(f"\n  Reviews 1 and 2 are exact duplicates.")
    print(f"  Tables identical: {tables[0] == tables[1]}")

    print("\n" + "=" * 65)
    print("  Hasher validation complete.")
    print("=" * 65)