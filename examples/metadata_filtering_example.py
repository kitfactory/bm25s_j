#!/usr/bin/env python3
"""
Example script demonstrating BM25S metadata filtering functionality.
BM25Sメタデータフィルタリング機能のデモンストレーション用サンプルスクリプトです。

This example shows how to:
このサンプルでは以下の方法を示します：
1. Create a BM25 index with metadata
   メタデータ付きBM25インデックスの作成
2. Perform filtered searches based on metadata
   メタデータに基づくフィルタリング検索の実行
3. Use different filter conditions
   異なるフィルタ条件の使用
"""

from bm25s import BM25
from bm25s.tokenization import tokenize


def main():
    """
    Main function demonstrating metadata filtering features.
    メタデータフィルタリング機能を実演するメイン関数です。
    """
    print("🔍 BM25S Metadata Filtering Example")
    print("=" * 50)
    
    # Sample corpus with diverse content
    # 多様なコンテンツを持つサンプルコーパス
    corpus = [
        "machine learning algorithms for data science",
        "legal contract document analysis",
        "deep learning neural networks tutorial",
        "corporate law and compliance guidelines",
        "python programming language basics",
        "medical research and clinical trials",
        "software engineering best practices",
        "healthcare data privacy regulations"
    ]
    
    # Metadata for each document
    # 各文書のメタデータ
    metadata = [
        {"category": "tech", "language": "en", "difficulty": "intermediate", "tags": ["ai", "data"]},
        {"category": "legal", "language": "en", "difficulty": "advanced", "tags": ["contract", "business"]},
        {"category": "tech", "language": "en", "difficulty": "advanced", "tags": ["ai", "neural"]},
        {"category": "legal", "language": "en", "difficulty": "intermediate", "tags": ["corporate", "compliance"]},
        {"category": "tech", "language": "en", "difficulty": "beginner", "tags": ["programming", "python"]},
        {"category": "medical", "language": "en", "difficulty": "advanced", "tags": ["research", "clinical"]},
        {"category": "tech", "language": "en", "difficulty": "intermediate", "tags": ["engineering", "software"]},
        {"category": "medical", "language": "en", "difficulty": "intermediate", "tags": ["privacy", "data"]}
    ]
    
    print(f"📚 Corpus: {len(corpus)} documents")
    print(f"🏷️  Metadata fields: {list(metadata[0].keys())}")
    print()
    
    # Tokenize and create BM25 index with metadata
    # トークン化してメタデータ付きBM25インデックスを作成
    print("🔧 Creating BM25 index with metadata...")
    tokenized_corpus = tokenize(corpus)
    
    bm25 = BM25()
    bm25.index(tokenized_corpus, metadata=metadata)
    print("✅ Index created successfully!")
    print()
    
    # Example 1: Filter by category
    # 例1: カテゴリでフィルタリング
    print("📋 Example 1: Filter by category='tech'")
    query = tokenize(["machine learning"])
    
    results = bm25.retrieve(
        query,
        k=len(corpus),  # Use corpus length to avoid k > available documents
        filter={"category": "tech"},
        return_metadata=True
    )
    
    print(f"Found {len(results.documents[0])} tech documents:")
    for i, (doc_idx, score, meta) in enumerate(zip(
        results.documents[0], results.scores[0], results.metadata[0]
    )):
        print(f"  {i+1}. Doc {doc_idx}: {corpus[doc_idx][:50]}...")
        print(f"      Score: {score:.4f}, Category: {meta['category']}, Tags: {meta['tags']}")
    print()
    
    # Example 2: Filter by multiple conditions (AND)
    # 例2: 複数条件でフィルタリング（AND）
    print("📋 Example 2: Filter by category='tech' AND difficulty='intermediate'")
    results = bm25.retrieve(
        query,
        k=len(corpus),
        filter={"category": "tech", "difficulty": "intermediate"},
        return_metadata=True
    )
    
    print(f"Found {len(results.documents[0])} intermediate tech documents:")
    for i, (doc_idx, score, meta) in enumerate(zip(
        results.documents[0], results.scores[0], results.metadata[0]
    )):
        print(f"  {i+1}. Doc {doc_idx}: {corpus[doc_idx][:50]}...")
        print(f"      Score: {score:.4f}, Difficulty: {meta['difficulty']}, Tags: {meta['tags']}")
    print()
    
    # Example 3: Filter by multiple values (OR)
    # 例3: 複数値でフィルタリング（OR）
    print("📋 Example 3: Filter by category in ['tech', 'medical']")
    query = tokenize(["data"])
    
    results = bm25.retrieve(
        query,
        k=len(corpus),
        filter={"category": ["tech", "medical"]},
        return_metadata=True
    )
    
    print(f"Found {len(results.documents[0])} tech or medical documents:")
    for i, (doc_idx, score, meta) in enumerate(zip(
        results.documents[0], results.scores[0], results.metadata[0]
    )):
        print(f"  {i+1}. Doc {doc_idx}: {corpus[doc_idx][:50]}...")
        print(f"      Score: {score:.4f}, Category: {meta['category']}, Tags: {meta['tags']}")
    print()
    
    # Example 4: Filter by list values (tags)
    # 例4: リスト値でフィルタリング（タグ）
    print("📋 Example 4: Filter by tag='ai'")
    query = tokenize(["algorithm"])
    
    results = bm25.retrieve(
        query,
        k=len(corpus),
        filter={"tags": "ai"},
        return_metadata=True
    )
    
    if len(results.documents) > 0 and len(results.documents[0]) > 0:
        print(f"Found {len(results.documents[0])} documents with 'ai' tag:")
        for i, (doc_idx, score, meta) in enumerate(zip(
            results.documents[0], results.scores[0], results.metadata[0]
        )):
            print(f"  {i+1}. Doc {doc_idx}: {corpus[doc_idx][:50]}...")
            print(f"      Score: {score:.4f}, Category: {meta['category']}, Tags: {meta['tags']}")
    else:
        print("Found 0 documents with 'ai' tag (query didn't match any tagged documents)")
    print()
    
    # Example 5: No matches
    # 例5: マッチなし
    print("📋 Example 5: Filter with no matches")
    results = bm25.retrieve(
        query,
        k=len(corpus),
        filter={"category": "nonexistent"},
        return_metadata=True
    )
    
    if len(results.documents) > 0 and len(results.documents[0]) > 0:
        print(f"Found {len(results.documents[0])} documents (expected: 0)")
    else:
        print("Found 0 documents (expected: 0) ✅")
    print()
    
    # Performance comparison
    # パフォーマンス比較
    print("⚡ Performance comparison")
    import time
    
    query = tokenize(["data science"])
    
    # Without filter
    # フィルタなし
    start_time = time.time()
    results_no_filter = bm25.retrieve(query, k=len(corpus))
    time_no_filter = time.time() - start_time
    
    # With filter
    # フィルタあり
    start_time = time.time()
    results_with_filter = bm25.retrieve(query, k=len(corpus), filter={"category": "tech"})
    time_with_filter = time.time() - start_time
    
    print(f"  Without filter: {len(results_no_filter.documents[0])} results in {time_no_filter:.4f}s")
    print(f"  With filter:    {len(results_with_filter.documents[0])} results in {time_with_filter:.4f}s")
    print()
    
    print("🎉 Metadata filtering example completed successfully!")


if __name__ == "__main__":
    main() 