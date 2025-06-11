#!/usr/bin/env python3
"""Debug script for metadata filtering operators"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import bm25s
import numpy as np


def main():
    print("🔍 Debug Metadata Filtering")
    
    # Simple test data
    corpus = ["Python programming", "JavaScript web"]
    metadata = [
        {"category": "programming", "score": 0.95},
        {"category": "web", "score": 0.85}
    ]
    
    bm25 = bm25s.BM25()
    bm25.index(corpus, metadata=metadata)
    
    print(f"Corpus: {corpus}")
    print(f"Metadata: {metadata}")
    print(f"Vocabulary: {bm25.vocab}")
    print(f"Field indices: {bm25.metadata_filter.field_indices}")
    
    # Test basic filter
    print("\n=== Testing basic filter ===")
    filter_result = bm25.metadata_filter.apply_filter({"category": "programming"})
    print(f"Filter result for category='programming': {filter_result}")
    
    # Test weight mask creation
    print("\n=== Testing weight mask ===")
    weight_mask = bm25.metadata_filter.create_weight_mask(filter_result, 2)
    print(f"Weight mask: {weight_mask}")
    
    # Test BM25 scoring without weight mask
    print("\n=== Testing BM25 scoring without weight mask ===")
    scores_no_mask = bm25.get_scores(["programming"])
    print(f"BM25 scores without weight mask: {scores_no_mask}")
    
    # Test BM25 scoring with weight mask
    print("\n=== Testing BM25 scoring with weight mask ===")
    scores = bm25.get_scores(["programming"], weight_mask=weight_mask)
    print(f"BM25 scores with weight mask: {scores}")
    
    # Check if "programming" is in vocabulary
    print("\n=== Checking vocabulary ===")
    token_ids = bm25.get_tokens_ids(["programming"])
    print(f"Token IDs for 'programming': {token_ids}")
    
    # Test with token that should exist
    print("\n=== Testing with existing token ===")
    try:
        if "python" in bm25.vocab:
            print("Testing with 'python' token")
            scores_python = bm25.get_scores(["python"], weight_mask=weight_mask)
            print(f"BM25 scores for 'python' with weight mask: {scores_python}")
        else:
            print("'python' not in vocabulary")
            print(f"Available tokens: {list(bm25.vocab.keys())[:10]}")
    except Exception as e:
        print(f"Error testing with python: {e}")
    
    # Test with smaller k value
    print("\n=== Testing top-k with k=1 ===")
    try:
        top_scores, top_indices = bm25._get_top_k_results(
            ["python"], k=1, weight_mask=weight_mask
        )
        print(f"Top scores: {top_scores}")
        print(f"Top indices: {top_indices}")
    except Exception as e:
        print(f"Error in top-k selection: {e}")


if __name__ == "__main__":
    main() 