# NLPrint: Amazon Review Authenticity Detector

**Course:** MSML606 - Data Structures and Algorithms, Spring 2026  
**Author:** Monark Dixit (UID: 122259645) | Group 13 
**Submission:** HW3 Extra Credit  
**Dataset:** McAuley-Lab/Amazon-Reviews-2023 (All_Beauty, 10,000+ records)

---

## What This Is

I built NLPrint because I wanted to see if hashing could solve a real, tangible problem rather than just exist as a theoretical concept in a textbook. Fake and copy-pasted reviews are genuinely everywhere on e-commerce platforms, and the interesting thing is that detecting them is fundamentally a similarity search problem. And similarity search at scale is exactly what hashing is good at.

NLPrint takes any Amazon review as input and finds the most suspiciously similar reviews from a pre-indexed dataset of 10,000+ beauty product reviews. It uses MinHash signatures and feature hashing to do this in milliseconds, without ever doing a brute-force pairwise comparison.

---

## How It Works

The pipeline has four stages:

**1. Preprocessing**  
Each review is stripped of HTML artifacts, unicode-normalized, emoji-tokenized, lowercased, and filtered for stopwords. The cleaned text is then broken into overlapping character trigrams called shingles. For example, the word "great" produces the shingles `gre`, `rea`, `eat`, `at `. These shingles are the raw material for everything that follows.

**2. Feature Hashing**  
The shingle set is mapped into a fixed-size integer array using a universal hash function I implemented from scratch. The universal hash function uses the multiply-add-mod scheme: `h(x) = ((a * x + b) mod p) mod table_size`, where `a` and `b` are randomly chosen coefficients and `p` is a Mersenne prime. This gives a compact, fixed-width representation of any review regardless of its length.

**3. MinHash Signatures**  
This was the hardest part to get right. MinHash works on a beautiful mathematical property: if you apply a random hash function to two sets and look at the minimum hash value in each, the probability that those two minimums are equal is exactly the Jaccard similarity of the sets. By doing this with k=100 independent hash functions, you get a signature vector of length 100 that encodes the similarity structure of the review without storing the review itself. The key implementation challenge was making the hash function deterministic across Python process restarts, which required replacing Python's built-in `hash()` with a custom FNV-1a implementation.

**4. Query and Lookup**  
At query time, the input review goes through the same pipeline and its signature is compared against every stored signature. The comparison is O(n * k) integer operations rather than O(n^2) text comparisons. On 10,000 reviews with k=100, this runs in under 100ms.

---

## Project Structure

```
nlprint/
├── src/
│   ├── __init__.py         # Package init and exports
│   ├── preprocessor.py     # HTML strip, unicode norm, shingle tokenizer
│   ├── hasher.py           # Universal hash function and feature hasher
│   ├── minhash.py          # MinHash signature generator
│   ├── indexer.py          # Index builder and persistence
│   ├── query.py            # Query engine and metadata flags
│   └── data_loader.py      # HuggingFace download and data audit
├── tests/
│   ├── __init__.py
│   └── test_preprocessor.py
├── data/                   # Downloaded dataset and saved index (gitignored)
├── assets/                 # Demo screenshots
├── app.py                  # Streamlit frontend
├── requirements.txt
├── AI_USAGE_DISCLOSURE.md
└── README.md
```

---

## Setup and Installation

**Requirements:** Python 3.11+, pip

**Step 1: Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/nlprint.git
cd nlprint
```

**Step 2: Create and activate a virtual environment**
```bash
# Windows
python -m venv venv --without-pip
venv\Scripts\activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Download the dataset and build the index**

On first run, the app handles this automatically. Or you can run it manually:
```bash
python src/data_loader.py
python src/indexer.py
```

The index build takes about 60 to 90 seconds and is saved to `data/signature_index.json`. Every subsequent run loads it from disk in under a second.

**Step 5: Launch the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Using the App

Paste any Amazon review into the input box and click **ANALYZE REVIEW**. The app returns the top-N most similar reviews from the dataset with:

- A similarity percentage badge (red = high similarity, amber = medium, green = low)
- The matched review text, title, rating, and date
- Metadata flags for suspicious signals:
  - **SHARED ACCOUNT** - the same user ID posted both reviews
  - **CLOSE TIMESTAMP** - reviews were submitted within 24 hours of each other
  - **PURCHASE MISMATCH** - one review is verified, the other is not

The sidebar shows the algorithm parameters (k, shingle size, hash table size, load factor) and the query latency for the last search.

---

## Running the Validation Scripts

Each module has its own validation entry point:

```bash
# Audit the raw dataset
python src/data_loader.py

# Spot-check preprocessing on 10 sample reviews
python src/preprocessor.py

# Validate the hash function and check collision rates
python src/hasher.py

# Validate MinHash accuracy against true Jaccard similarity
python src/minhash.py

# Build the index and benchmark throughput
python src/indexer.py

# Benchmark query latency
python src/query.py
```

---

## Algorithm Parameters

| Parameter | Default | Notes |
|---|---|---|
| Shingle size | 3 (trigrams) | Character n-gram length |
| MinHash functions (k) | 100 | Higher k = more accurate, slower |
| Hash table size | 1024 buckets | Affects load factor |
| Min similarity threshold | 0.01 | Results below this are filtered |
| Timestamp proximity window | 24 hours | For suspicious flag detection |

---

## Dependencies

```
datasets
streamlit
pandas
numpy
nltk
unicodedata2
langdetect
huggingface_hub
```

All dependencies are in `requirements.txt`. No external similarity search libraries are used. The MinHash and feature hashing implementations are written entirely from scratch.

---

## AI Usage Disclosure

See `AI_USAGE_DISCLOSURE.md` for a full account of how AI assistance was used in this project.

---

## Course Context

This project was built for MSML606 HW3 Extra Credit. The goal was to demonstrate a practical, real-world application of hashing where the algorithm is not just a theoretical construct but the reason the application can operate at scale. Brute-force pairwise comparison across 10,000+ records in real time is not viable. Hashing makes it possible.
