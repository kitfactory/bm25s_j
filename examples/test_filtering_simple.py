#!/usr/bin/env python3
"""Simple test for advanced metadata filtering functionality"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import bm25s


def main():
    print("🎯 Testing Advanced Metadata Filtering")
    print("=" * 50)
    
    # Create test corpus
    corpus = [
        "Python programming tutorial for beginners",
        "Advanced JavaScript web development",
        "Machine learning with Python AI",
        "HTML and CSS styling guide",
        "Data science Python analysis"
    ]
    
    metadata = [
        {"category": "programming", "difficulty": "beginner", "score": 0.85, "language": "python"},
        {"category": "web-development", "difficulty": "advanced", "score": 0.92, "language": "javascript"},
        {"category": "ai", "difficulty": "intermediate", "score": 0.95, "language": "python"},
        {"category": "web-development", "difficulty": "beginner", "score": 0.75, "language": "html"},
        {"category": "data-science", "difficulty": "intermediate", "score": 0.90, "language": "python"}
    ]
    
    # Create BM25 instance
    bm25 = bm25s.BM25()
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=None)
    bm25.index(corpus_tokens, metadata=metadata)
    
    print(f"✅ Indexed {len(corpus)} documents")
    print(f"✅ Vocabulary size: {len(bm25.vocab_dict)}")
    
    # Test 1: Comparison operators
    print("\n🔢 Test 1: Comparison Operators")
    results = bm25.retrieve([["python"]], k=3, filter={"score": {"$gte": 0.90}})
    print(f"   High-score Python content: {len(results.documents[0])} results")
    
    # Test 2: List operators
    print("\n📋 Test 2: List Operators")
    results = bm25.retrieve([["programming"]], k=3, filter={"language": {"$in": ["python", "javascript"]}})
    print(f"   Python/JS content: {len(results.documents[0])} results")
    
    # Test 3: Exists operator
    print("\n🔍 Test 3: Exists Operator")
    results = bm25.retrieve([["development"]], k=5, filter={"category": {"$exists": True}})
    print(f"   Documents with category: {len(results.documents[0])} results")
    
    # Test 4: Logical operators
    print("\n🔀 Test 4: Logical Operators")
    results = bm25.retrieve([["python"]], k=3, filter={
        "$or": [
            {"category": "programming"},
            {"difficulty": "advanced"}
        ]
    })
    print(f"   Programming OR advanced: {len(results.documents[0])} results")
    
    # Test 5: Complex nested conditions
    print("\n🌟 Test 5: Complex Conditions")
    results = bm25.retrieve([["python"]], k=3, filter={
        "$and": [
            {"score": {"$gte": 0.85}},
            {"$or": [
                {"category": "programming"},
                {"category": "ai"}
            ]}
        ]
    })
    print(f"   High-score programming/AI: {len(results.documents[0])} results")
    
    print("\n✅ All filtering tests completed successfully!")
    print("✅ すべてのフィルタリングテストが正常に完了しました！")


if __name__ == "__main__":
    main() 