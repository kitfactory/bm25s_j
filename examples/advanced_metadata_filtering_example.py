#!/usr/bin/env python3
"""
Advanced Metadata Filtering Example for BM25s-j

This example demonstrates advanced metadata filtering capabilities including:
- Comparison operators ($gt, $gte, $lt, $lte, $ne)
- List operators ($in, $nin)
- Existence operators ($exists)
- Regular expression operators ($regex)
- Logical operators ($or, $and, $not)
- Complex nested conditions

このサンプルは以下の高度なメタデータフィルタリング機能を実演します：
- 比較演算子 ($gt, $gte, $lt, $lte, $ne)
- リスト演算子 ($in, $nin)
- 存在演算子 ($exists)  
- 正規表現演算子 ($regex)
- 論理演算子 ($or, $and, $not)
- 複雑な入れ子条件
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import bm25s
import time


def create_sample_data():
    """
    Create sample corpus with rich metadata for demonstration.
    実演用のリッチなメタデータを持つサンプルコーパスを作成します。
    """
    corpus = [
        "Python is a high-level programming language with excellent machine learning libraries",
        "JavaScript frameworks like React and Vue make web development efficient and modern", 
        "Java enterprise applications provide robust backend solutions for large organizations",
        "HTML5 semantic markup improves web accessibility and search engine optimization",
        "CSS Grid and Flexbox revolutionized responsive web design and layout techniques",
        "Docker containers enable consistent deployment across different development environments",
        "Machine learning algorithms can predict customer behavior patterns effectively",
        "RESTful APIs facilitate seamless integration between microservices architectures",
        "Database indexing strategies significantly improve query performance in large datasets",
        "Cloud computing platforms provide scalable infrastructure for modern applications"
    ]
    
    metadata = [
        {
            "category": "programming", 
            "difficulty": "beginner", 
            "score": 0.95,
            "tags": ["python", "ai", "data-science", "machine-learning"],
            "date": "2024-01-01",
            "language": "python",
            "author": "tech_expert",
            "views": 15000
        },
        {
            "category": "web-development", 
            "difficulty": "intermediate", 
            "score": 0.88,
            "tags": ["javascript", "react", "vue", "frontend"],
            "date": "2024-01-15",
            "language": "javascript",  
            "author": "web_dev",
            "views": 12000
        },
        {
            "category": "programming", 
            "difficulty": "advanced", 
            "score": 0.92,
            "tags": ["java", "enterprise", "backend", "microservices"],
            "date": "2024-02-01",
            "language": "java",
            "author": "enterprise_dev",
            "views": 8000
        },
        {
            "category": "web-development", 
            "difficulty": "beginner", 
            "score": 0.75,
            "tags": ["html", "html5", "semantic", "accessibility"],
            "date": "2024-01-10",
            "language": "html",
            "author": "web_designer",
            "views": 20000
        },
        {
            "category": "web-development", 
            "difficulty": "beginner", 
            "score": 0.80,
            "tags": ["css", "grid", "flexbox", "responsive"],
            "date": "2024-01-20",
            "language": "css",
            "author": "ui_specialist",
            "views": 18000
        },
        {
            "category": "devops", 
            "difficulty": "intermediate", 
            "score": 0.90,
            "tags": ["docker", "containers", "deployment", "devops"],
            "date": "2024-02-15",
            "language": "docker",
            "author": "devops_engineer",
            "views": 10000
        },
        {
            "category": "data-science", 
            "difficulty": "advanced", 
            "score": 0.94,
            "tags": ["machine-learning", "ai", "prediction", "algorithms"],
            "date": "2024-03-01",
            "language": "python",
            "author": "data_scientist",
            "views": 25000
        },
        {
            "category": "backend", 
            "difficulty": "intermediate", 
            "score": 0.85,
            "tags": ["api", "rest", "microservices", "integration"],
            "date": "2024-02-20",
            "language": "api",
            "author": "api_architect",
            "views": 14000
        },
        {
            "category": "database", 
            "difficulty": "advanced", 
            "score": 0.91,
            "tags": ["database", "indexing", "performance", "optimization"],
            "date": "2024-03-10",
            "language": "sql",
            "author": "db_admin",
            "views": 16000
        },
        {
            "category": "cloud", 
            "difficulty": "intermediate", 
            "score": 0.87,
            "tags": ["cloud", "scalability", "infrastructure", "aws"],
            "date": "2024-03-15",
            "language": "cloud",
            "author": "cloud_architect",
            "views": 22000
        }
    ]
    
    return corpus, metadata


def demonstrate_comparison_operators(bm25):
    """
    Demonstrate comparison operators.
    比較演算子を実演します。
    """
    print("\n=== 🔢 Comparison Operators / 比較演算子 ===")
    
    # High score content
    print("\n1. High-quality content (score >= 0.90):")
    print("   高品質コンテンツ (score >= 0.90):")
    results = bm25.retrieve([["technology"]], k=5, filter={"score": {"$gte": 0.90}}, return_metadata=True)
    print(f"   Found {len(results.documents[0])} high-quality documents")
    print(f"   {len(results.documents[0])}件の高品質文書が見つかりました")
    if len(results.documents[0]) > 0:
        for i, (idx, score, meta) in enumerate(zip(results.documents[0], results.scores[0], results.metadata[0])):
            print(f"   [{i+1}] Score: {meta['score']:.2f} | {meta['category']} | {meta['author']}")
    
    # Not equal filter
    print("\n2. Non-programming content:")
    print("   プログラミング以外のコンテンツ:")
    results = bm25.retrieve([["technology"]], k=4, filter={"category": {"$ne": "programming"}}, return_metadata=True)
    print(f"   Found {len(results.documents[0])} non-programming documents")
    print(f"   {len(results.documents[0])}件のプログラミング以外の文書が見つかりました")
    if len(results.documents[0]) > 0:
        for i, (idx, score, meta) in enumerate(zip(results.documents[0], results.scores[0], results.metadata[0])):
            print(f"   [{i+1}] {meta['category']} | {meta['language']} | Views: {meta['views']}")


def demonstrate_list_operators(bm25):
    """
    Demonstrate list operators ($in, $nin).
    リスト演算子を実演します。
    """
    print("\n=== 📋 List Operators / リスト演算子 ===")
    
    # $in operator
    print("\n1. Web technologies (language in ['javascript', 'html', 'css']):")
    print("   Web技術 (language in ['javascript', 'html', 'css']):")
    results = bm25.retrieve([["web"]], k=5, filter={"language": {"$in": ["javascript", "html", "css"]}}, return_metadata=True)
    print(f"   Found {len(results.documents[0])} web technology documents")
    print(f"   {len(results.documents[0])}件のWeb技術文書が見つかりました")
    if len(results.documents[0]) > 0:
        for i, (idx, score, meta) in enumerate(zip(results.documents[0], results.scores[0], results.metadata[0])):
            print(f"   [{i+1}] {meta['language'].upper()} | {meta['difficulty']} | {meta['category']}")
    
    # $nin operator
    print("\n2. Non-beginner content (difficulty not in ['beginner']):")
    print("   初心者レベル以外のコンテンツ (difficulty not in ['beginner']):")
    results = bm25.retrieve([["programming"]], k=4, filter={"difficulty": {"$nin": ["beginner"]}}, return_metadata=True)
    print(f"   Found {len(results.documents[0])} advanced documents")
    print(f"   {len(results.documents[0])}件の上級文書が見つかりました")
    if len(results.documents[0]) > 0:
        for i, (idx, score, meta) in enumerate(zip(results.documents[0], results.scores[0], results.metadata[0])):
            print(f"   [{i+1}] {meta['difficulty'].title()} | {meta['category']} | {meta['author']}")


def demonstrate_existence_operator(bm25):
    """
    Demonstrate existence operator ($exists).
    存在演算子を実演します。
    """  
    print("\n=== 🔍 Existence Operator / 存在演算子 ===")
    
    # Field exists
    print("\n1. Documents with author information (author exists):")
    print("   著者情報があるドキュメント (author exists):")
    results = bm25.retrieve("development", k=5, filter={"author": {"$exists": True}}, return_metadata=True)
    print(f"   Found {len(results.indices)} documents (all have author field)")
    print(f"   {len(results.indices)}件のドキュメントが見つかりました（すべて著者フィールドあり）")
    
    # Field doesn't exist
    print("\n2. Documents without publisher field (publisher doesn't exist):")
    print("   出版社フィールドがないドキュメント (publisher doesn't exist):")
    results = bm25.retrieve("technology", k=3, filter={"publisher": {"$exists": False}}, return_metadata=True)
    print(f"   Found {len(results.indices)} documents (none have publisher field)")
    print(f"   {len(results.indices)}件のドキュメントが見つかりました（出版社フィールドなし）")


def demonstrate_regex_operator(bm25):
    """
    Demonstrate regular expression operator ($regex).
    正規表現演算子を実演します。
    """
    print("\n=== 🔤 Regular Expression Operator / 正規表現演算子 ===")
    
    # Find AI/ML content
    results = bm25.retrieve("artificial intelligence", k=3, filter={"tags": {"$regex": ".*(machine|ai).*"}}, return_metadata=True)
    print(f"\n1. AI/ML content (tags matching '.*machine.*|.*ai.*'): {len(results.indices)} results")
    print(f"   AI/ML コンテンツ: {len(results.indices)}件")
    for i, (idx, score, meta) in enumerate(zip(results.indices, results.scores, results.metadata)):
        ai_tags = [tag for tag in meta['tags'] if 'machine' in tag or 'ai' in tag]
        print(f"   [{i+1}] {meta['category']} | AI tags: {ai_tags} | Score: {meta['score']}")


def demonstrate_logical_operators(bm25):
    """
    Demonstrate logical operators ($or, $and, $not).
    論理演算子を実演します。
    """
    print("\n=== 🔀 Logical Operators / 論理演算子 ===")
    
    # OR operator
    results = bm25.retrieve("development", k=4, filter={
        "$or": [
            {"category": "web-development"},
            {"score": {"$gte": 0.92}}
        ]
    }, return_metadata=True)
    print(f"\n1. Web development OR high-score content: {len(results.indices)} results")
    print(f"   Web開発または高スコアコンテンツ: {len(results.indices)}件")
    for i, (idx, score, meta) in enumerate(zip(results.indices, results.scores, results.metadata)):
        condition = "web-dev" if meta['category'] == "web-development" else f"high-score({meta['score']})"
        print(f"   [{i+1}] {condition} | {meta['difficulty']} | {meta['language']}")
    
    # NOT operator
    results = bm25.retrieve("technology", k=4, filter={
        "$not": {"difficulty": "beginner"}
    }, return_metadata=True)
    print(f"\n2. Non-beginner content: {len(results.indices)} results")
    print(f"   初心者向けでないコンテンツ: {len(results.indices)}件")
    for i, (idx, score, meta) in enumerate(zip(results.indices, results.scores, results.metadata)):
        print(f"   [{i+1}] {meta['difficulty'].title()} | {meta['category']} | Views: {meta['views']}")


def demonstrate_complex_conditions(bm25):
    """
    Demonstrate complex nested conditions.
    複雑な入れ子条件を実演します。
    """
    print("\n=== 🌟 Complex Nested Conditions / 複雑な入れ子条件 ===")
    
    # Complex condition: (High engagement AND intermediate+) OR (AI/ML content)
    print("\n1. Popular advanced content OR AI/ML content:")
    print("   人気の高度コンテンツまたはAI/MLコンテンツ:")
    print("   Condition: (views >= 15000 AND difficulty != 'beginner') OR tags contains 'ai'")
    print("   条件: (views >= 15000 AND difficulty != 'beginner') OR tags contains 'ai'")
    
    results = bm25.retrieve("technology machine learning", k=5, filter={
        "$or": [
            {
                "$and": [
                    {"views": {"$gte": 15000}},
                    {"difficulty": {"$ne": "beginner"}}
                ]
            },
            {"tags": {"$regex": ".*ai.*"}}
        ]
    }, return_metadata=True)
    
    for i, (idx, score, meta) in enumerate(zip(results.indices, results.scores, results.metadata)):
        condition1 = f"popular({meta['views']})" if meta['views'] >= 15000 and meta['difficulty'] != 'beginner' else ""
        condition2 = "AI-content" if any('ai' in tag for tag in meta['tags']) else ""
        match_reason = condition1 or condition2 or "both"
        print(f"   [{i+1}] {match_reason} | {meta['category']} | {meta['difficulty']} | Score: {meta['score']:.2f}")
    
    # Another complex condition
    print("\n2. High-quality technical content with specific criteria:")
    print("   特定条件を満たす高品質技術コンテンツ:")
    print("   Condition: score >= 0.85 AND (category='programming' OR language='python') AND NOT beginner")
    print("   条件: score >= 0.85 AND (category='programming' OR language='python') AND NOT beginner")
    
    results = bm25.retrieve("programming python development", k=4, filter={
        "$and": [
            {"score": {"$gte": 0.85}},
            {
                "$or": [
                    {"category": "programming"},
                    {"language": "python"}
                ]
            },
            {"$not": {"difficulty": "beginner"}}
        ]
    }, return_metadata=True)
    
    for i, (idx, score, meta) in enumerate(zip(results.indices, results.scores, results.metadata)):
        cat_reason = "programming" if meta['category'] == "programming" else f"python-lang"
        print(f"   [{i+1}] Score: {meta['score']:.2f} | {cat_reason} | {meta['difficulty']} | {meta['author']}")


def demonstrate_performance_comparison(bm25, corpus, metadata):
    """
    Demonstrate performance comparison between filtered and unfiltered search.
    フィルタありとフィルタなしの検索のパフォーマンス比較を実演します。
    """
    print("\n=== ⚡ Performance Comparison / パフォーマンス比較 ===")
    
    query = "web development programming"
    k = 5
    
    # Unfiltered search
    start_time = time.time()
    results_unfiltered = bm25.retrieve(query, k=10)  # Get more results to simulate post-filtering
    unfiltered_time = time.time() - start_time
    
    # Filtered search  
    start_time = time.time()
    results_filtered = bm25.retrieve(query, k=k, filter={"category": {"$in": ["web-development", "programming"]}})
    filtered_time = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"   Unfiltered search: {len(results_unfiltered.indices)} results in {unfiltered_time*1000:.2f}ms")
    print(f"   フィルタなし検索: {len(results_unfiltered.indices)}件の結果を{unfiltered_time*1000:.2f}msで取得")
    print(f"   Filtered search: {len(results_filtered.indices)} results in {filtered_time*1000:.2f}ms") 
    print(f"   フィルタあり検索: {len(results_filtered.indices)}件の結果を{filtered_time*1000:.2f}msで取得")
    
    if filtered_time < unfiltered_time:
        improvement = ((unfiltered_time - filtered_time) / unfiltered_time) * 100
        print(f"   🚀 Filtering improved performance by {improvement:.1f}%")
        print(f"   🚀 フィルタリングによりパフォーマンスが{improvement:.1f}%向上")
    
    print(f"\n📈 Memory efficiency:")
    print(f"   メモリ効率:")
    print(f"   - Early document filtering reduces processing overhead")
    print(f"   - 早期文書フィルタリングにより処理オーバーヘッドを削減")
    print(f"   - BM25 scoring only applied to relevant documents")  
    print(f"   - BM25スコアリングは関連文書のみに適用")


def main():
    """
    Main demonstration function.
    メイン実演関数です。
    """
    print("🔍 Advanced Metadata Filtering Example for BM25s-j")
    print("🔍 BM25s-j 高度なメタデータフィルタリング実演")
    print("=" * 60)
    
    # Create sample data
    print("\n📚 Creating sample corpus with rich metadata...")
    print("📚 リッチメタデータを持つサンプルコーパスを作成中...")
    corpus, metadata = create_sample_data()
    
    # Create and index BM25 with metadata
    # メタデータ付きでBM25を作成・インデックス化
    bm25 = bm25s.BM25()
    
    # Tokenize corpus properly using word-level tokenization
    # 適切な単語レベルのトークン化を使用してコーパスをトークン化
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=None)
    bm25.index(corpus_tokens, metadata=metadata)
    
    # Run demonstrations
    demonstrate_comparison_operators(bm25)
    demonstrate_list_operators(bm25)  
    demonstrate_existence_operator(bm25)
    demonstrate_regex_operator(bm25)
    demonstrate_logical_operators(bm25)
    demonstrate_complex_conditions(bm25)
    demonstrate_performance_comparison(bm25, corpus, metadata)
    
    print("\n" + "=" * 60)
    print("🎉 Advanced filtering demonstration completed!")
    print("🎉 高度なフィルタリング実演が完了しました！")
    print("\n💡 Key benefits of advanced metadata filtering:")
    print("💡 高度なメタデータフィルタリングの主要メリット:")
    print("   • Precise content targeting with complex conditions")
    print("   • 複雑な条件による精密なコンテンツターゲティング")
    print("   • Improved search performance through early filtering")
    print("   • 早期フィルタリングによる検索パフォーマンス向上")
    print("   • Flexible query composition for diverse use cases")
    print("   • 多様なユースケースに対応する柔軟なクエリ構成")
    print("   • Enterprise-ready metadata management")
    print("   • エンタープライズ対応のメタデータ管理")
    print("\n💡 Supported advanced operators:")
    print("💡 サポートされる高度な演算子:")
    print("   • Comparison: $gt, $gte, $lt, $lte, $ne")
    print("   • 比較: $gt, $gte, $lt, $lte, $ne")
    print("   • List: $in, $nin")
    print("   • リスト: $in, $nin")
    print("   • Existence: $exists")
    print("   • 存在: $exists")
    print("   • Regex: $regex")
    print("   • 正規表現: $regex")
    print("   • Logical: $or, $and, $not")
    print("   • 論理: $or, $and, $not")


if __name__ == "__main__":
    main() 