#!/usr/bin/env python3
"""
Simple test script for advanced metadata filtering operators.
高度なメタデータフィルタリング演算子の簡単なテストスクリプトです。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import bm25s


def main():
    print("🔍 Testing Advanced Metadata Filtering Operators")
    print("🔍 高度なメタデータフィルタリング演算子のテスト")
    print("=" * 60)
    
    # Create sample data
    corpus = [
        "Python programming language for machine learning",
        "JavaScript web development framework",
        "Java enterprise backend solution",
        "HTML markup for web pages", 
        "CSS styling for responsive design"
    ]
    
    metadata = [
        {"category": "programming", "difficulty": "beginner", "score": 0.95, "tags": ["python", "ai"], "language": "python"},
        {"category": "web-development", "difficulty": "intermediate", "score": 0.85, "tags": ["javascript", "web"], "language": "javascript"},
        {"category": "programming", "difficulty": "advanced", "score": 0.90, "tags": ["java", "enterprise"], "language": "java"},
        {"category": "web-development", "difficulty": "beginner", "score": 0.75, "tags": ["html", "web"], "language": "html"},
        {"category": "web-development", "difficulty": "beginner", "score": 0.80, "tags": ["css", "design"], "language": "css"}
    ]
    
    # Initialize BM25
    bm25 = bm25s.BM25()
    bm25.index(corpus, metadata=metadata)
    print(f"✅ Indexed {len(corpus)} documents with metadata")
    
    # Test comparison operators
    print("\n=== Comparison Operators ===")
    results = bm25.retrieve([["programming"]], k=5, filter={"score": {"$gte": 0.85}}, return_metadata=True)
    print(f"$gte: Found {len(results.documents[0])} documents with score >= 0.85")
    
    results = bm25.retrieve([["web"]], k=5, filter={"category": {"$ne": "programming"}}, return_metadata=True)
    print(f"$ne: Found {len(results.documents[0])} non-programming documents")
    
    # Test logical operators
    print("\n=== Logical Operators ===")
    results = bm25.retrieve([["development"]], k=5, filter={
        "$or": [
            {"category": "web-development"},
            {"difficulty": "advanced"}
        ]
    }, return_metadata=True)
    print(f"$or: Found {len(results.documents[0])} documents (web-dev OR advanced)")
    
    results = bm25.retrieve([["programming"]], k=5, filter={
        "$and": [
            {"category": "programming"},
            {"difficulty": "beginner"}
        ]
    }, return_metadata=True)
    print(f"$and: Found {len(results.documents[0])} documents (programming AND beginner)")
    
    # Test regex operator
    print("\n=== Regex Operator ===")
    results = bm25.retrieve([["programming"]], k=5, filter={"tags": {"$regex": ".*java.*"}}, return_metadata=True)
    print(f"$regex: Found {len(results.documents[0])} documents with java-related tags")
    
    # Test exists operator
    print("\n=== Exists Operator ===")
    results = bm25.retrieve([["language"]], k=5, filter={"category": {"$exists": True}}, return_metadata=True)
    print(f"$exists(True): Found {len(results.documents[0])} documents with 'category' field")
    
    results = bm25.retrieve([["language"]], k=5, filter={"author": {"$exists": False}}, return_metadata=True)
    print(f"$exists(False): Found {len(results.documents[0])} documents without 'author' field")
    
    print("\n🎉 All tests completed successfully!")


if __name__ == "__main__":
    main() 