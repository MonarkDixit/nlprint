# src/data_loader.py
# NLPrint: Amazon Review Authenticity Detector
# Author: Monark Dixit (UID: 122259645)
# Course: MSML606, Spring 2026

#PURPOSE:
#This module handles two responsibilities:
#1. Downloading the All_Beauty split from the HuggingFace Amazon Reviews
#      2023 dataset and saving it locally as a JSONL file.
#2. Auditing the raw data to surface quality issues before any record
#      enters the hashing pipeline. This audit is required for Phase 1.

#USAGE:
#Run directly to download and audit:
#  python src/data_loader.py

#Import in other modules to stream cleaned records:
#  from src.data_loader import load_records


import json
import os
import time
from datetime import datetime

#datasets is the HuggingFace library for loading Amazon Reviews 2023
from huggingface_hub import hf_hub_download
from streamlit import empty



#CONSTANTS
#These values control which dataset split we use and where we save it.
#Changing CATEGORY lets you swap to a different Amazon product category.


CATEGORY = "All_Beauty"          #Amazon product category to download
MAX_RECORDS = 10000              #Cap at 10,000 records as per project scope
DATA_DIR = "data"                #Folder where the JSONL file will be saved
JSONL_FILENAME = "all_beauty_reviews.jsonl"  # Output file name
JSONL_PATH = os.path.join(DATA_DIR, JSONL_FILENAME)

#These are the metadata fields we care about per the project proposal.
#Any field missing from a record will be flagged during the audit.
EXPECTED_FIELDS = [
    "text",
    "rating",
    "title",
    "user_id",
    "timestamp",
    "verified_purchase",
    "helpful_vote",
]


#download_dataset()
#Downloads the All_Beauty split from HuggingFace and saves it as JSONL.
#If the file already exists locally, the download is skipped to save time.

def download_dataset():         #Downloads the All_Beauty category from McAuley-Lab/Amazon-Reviews-2023 directly as a JSONL file using huggingface_hub, then saves up to MAX_RECORDS records locally.

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(JSONL_PATH):
        print(f"[INFO] Dataset already exists at '{JSONL_PATH}'. Skipping download.")
        return JSONL_PATH

    print(f"[INFO] Downloading '{CATEGORY}' reviews from HuggingFace...")
    print(f"[INFO] This may take a few minutes depending on your connection.\n")

    #huggingface_hub downloads the raw file directly from the dataset repo,
    #bypassing the loading script issue entirely
    from huggingface_hub import hf_hub_download

    start_time = time.time()

    raw_file = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=f"raw/review_categories/{CATEGORY}.jsonl",
        repo_type="dataset",
    )

    elapsed = time.time() - start_time
    print(f"[INFO] File downloaded in {elapsed:.1f}s. Now saving first {MAX_RECORDS:,} records...\n")

    records_written = 0
    with open(raw_file, "r", encoding="utf-8") as src, \
         open(JSONL_PATH, "w", encoding="utf-8") as dst:
        for line in src:
            if records_written >= MAX_RECORDS:
                break
            line = line.strip()
            if line:
                dst.write(line + "\n")
                records_written += 1

    print(f"[INFO] Saved {records_written:,} records to '{JSONL_PATH}'.")
    return JSONL_PATH


#audit_dataset()
#Reads the saved JSONL file and prints a detailed quality report.
#This satisfies the Phase 1 requirement to audit raw data before processing.

def audit_dataset():        #Reads the local JSONL file and prints a full audit reporting data quality issues, missing fields, rating distribution, and sample records.

    if not os.path.exists(JSONL_PATH):
        print(f"[ERROR] No dataset found at '{JSONL_PATH}'. Run download_dataset() first.")
        return {}

    print("=" * 65)
    print("  NLPrint: Data Audit Report")
    print(f"  File: {JSONL_PATH}")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    #Counters and collectors
    total_records = 0
    missing_text = 0          #Records where 'text' is absent or empty
    empty_text = 0            #Records where 'text' is whitespace only
    missing_fields = {f: 0 for f in EXPECTED_FIELDS}  #Per-field missing counts
    rating_dist = {}          #How many reviews per star rating
    verified_count = 0        #Records where verified_purchase is True
    unverified_count = 0      #Records where verified_purchase is False
    timestamps = []           #Collect all timestamps to find range
    has_unicode_issues = 0    #Records with non-ASCII characters in text
    sample_records = []       #First 3 records for display

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                #Skip any malformed lines
                continue

            total_records += 1

            #Collect first 3 records as samples
            if total_records <= 3:
                sample_records.append(record)

            #Check for missing or empty text field
            text_val = record.get("text", None)
            if text_val is None:
                missing_text += 1
            elif str(text_val).strip() == "":
                empty_text += 1

            #Check each expected metadata field
            for field in EXPECTED_FIELDS:
                if field not in record or record[field] is None:
                    missing_fields[field] += 1

            #Rating distribution 
            rating = record.get("rating", None)
            if rating is not None:
                rating_key = str(int(float(rating))) + " stars"
                rating_dist[rating_key] = rating_dist.get(rating_key, 0) + 1

            #Verified purchase breakdown 
            vp = record.get("verified_purchase", None)
            if vp is True:
                verified_count += 1
            elif vp is False:
                unverified_count += 1

            #Timestamp collection
            ts = record.get("timestamp", None)
            if ts is not None:
                try:
                    timestamps.append(int(ts))
                except (ValueError, TypeError):
                    pass

            #Unicode / encoding check
            #Reviews with non-ASCII characters may have emoji or non-English text.
            #We flag them here; the preprocessor will handle them properly.
            if text_val:
                try:
                    str(text_val).encode("ascii")
                except UnicodeEncodeError:
                    has_unicode_issues += 1

    #Print the report

    print(f"\n  RECORD COUNTS")
    print(f"  {'Total records:':<35} {total_records:,}")
    print(f"  {'Missing text field (null):':<35} {missing_text:,}")
    print(f"  {'Empty text field (whitespace):':<35} {empty_text:,}")
    print(f"  {'Records with unicode/emoji:':<35} {has_unicode_issues:,}")

    print(f"\n  MISSING FIELD COUNTS (per field)")
    for field, count in missing_fields.items():
        pct = (count / total_records * 100) if total_records > 0 else 0
        print(f"  {'  ' + field + ':':<35} {count:,}  ({pct:.1f}%)")

    print(f"\n  RATING DISTRIBUTION")
    for star in sorted(rating_dist.keys()):
        bar = "#" * (rating_dist[star] // 100)
        print(f"  {'  ' + star + ':':<15} {rating_dist[star]:>6,}  {bar}")

    print(f"\n  VERIFIED PURCHASE BREAKDOWN")
    print(f"  {'  Verified:':<35} {verified_count:,}")
    print(f"  {'  Not verified:':<35} {unverified_count:,}")

    if timestamps:
        earliest = datetime.utcfromtimestamp(min(timestamps) / 1000).strftime("%Y-%m-%d")
        latest = datetime.utcfromtimestamp(max(timestamps) / 1000).strftime("%Y-%m-%d")
        print(f"\n  TIMESTAMP RANGE")
        print(f"  {'  Earliest review:':<35} {earliest}")
        print(f"  {'  Latest review:':<35} {latest}")

    print(f"\n  SAMPLE RECORDS (first 3)")
    for i, rec in enumerate(sample_records, 1):
        print(f"\n  --- Record {i} ---")
        for field in EXPECTED_FIELDS:
            val = rec.get(field, "[MISSING]")
            #Truncate long text for display
            if isinstance(val, str) and len(val) > 80:
                val = val[:80] + "..."
            print(f"  {field}: {val}")

    print("\n" + "=" * 65)
    print("  Audit complete.")
    print("=" * 65)

    #Return a summary dict so other modules can use audit results if needed
    summary = {
        "total_records": total_records,
        "missing_text": missing_text,
        "empty_text": empty_text,
        "has_unicode_issues": has_unicode_issues,
        "missing_fields": missing_fields,
        "rating_distribution": rating_dist,
        "verified_count": verified_count,
        "unverified_count": unverified_count,
    }
    return summary


#load_records()
#A generator that yields cleaned, structured record dicts one at a time.
#Used by the indexer and preprocessor to stream through the dataset without
#loading everything into memory at once.

def load_records(skip_empty_text=True):         #Generator that reads the local JSONL file and yields one record dict at a time. Records with missing or empty text are skipped by default
                                                #since they cannot enter the hashing pipeline.

    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{JSONL_PATH}'. "
            f"Run download_dataset() first."
        )

    skipped = 0
    yielded = 0

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            #Skip records with no usable text before they reach the pipeline
            text_val = record.get("text", None)
            if skip_empty_text and (text_val is None or str(text_val).strip() == ""):
                skipped += 1
                continue

            yielded += 1
            yield record

    print(f"[INFO] load_records complete: {yielded:,} yielded, {skipped:,} skipped.")


#ENTRY POINT
#Running this file directly triggers the download and audit in sequence.

if __name__ == "__main__":
    print("[STEP 1] Downloading dataset...")
    download_dataset()

    print("\n[STEP 2] Running audit...")
    audit_dataset()