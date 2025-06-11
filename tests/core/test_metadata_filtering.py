#!/usr/bin/env python3

"""
Tests for metadata filtering functionality in BM25s
BM25sのメタデータフィルタリング機能のテスト
"""

import unittest
import numpy as np
from bm25s import BM25
from bm25s.filtering import MetadataFilter, validate_metadata
import bm25s


class TestMetadataFilter:
    """
    Test class for MetadataFilter functionality.
    MetadataFilter機能のテストクラスです。
    """
    
    def test_metadata_filter_initialization(self):
        """
        Test MetadataFilter initialization with valid metadata.
        有効なメタデータでのMetadataFilter初期化をテストします。
        """
        metadata = [
            {"category": "tech", "language": "ja", "difficulty": "beginner"},
            {"category": "legal", "language": "ja", "difficulty": "advanced"},
            {"category": "tech", "language": "en", "difficulty": "intermediate"},
        ]
        
        filter_engine = MetadataFilter(metadata)
        assert filter_engine.metadata == metadata
        assert len(filter_engine.field_indices) == 3  # category, language, difficulty
        assert "category" in filter_engine.field_indices
        assert "language" in filter_engine.field_indices
        assert "difficulty" in filter_engine.field_indices
    
    def test_simple_equality_filter(self):
        """
        Test simple equality filtering.
        単純な等価フィルタリングをテストします。
        """
        metadata = [
            {"category": "tech", "language": "ja"},
            {"category": "legal", "language": "ja"},
            {"category": "tech", "language": "en"},
            {"category": "legal", "language": "en"},
        ]
        
        filter_engine = MetadataFilter(metadata)
        
        # Filter by category
        # カテゴリでフィルタリング
        tech_docs = filter_engine.apply_filter({"category": "tech"})
        expected_tech = np.array([0, 2], dtype=np.int32)
        np.testing.assert_array_equal(tech_docs, expected_tech)
        
        # Filter by language
        # 言語でフィルタリング
        ja_docs = filter_engine.apply_filter({"language": "ja"})
        expected_ja = np.array([0, 1], dtype=np.int32)
        np.testing.assert_array_equal(ja_docs, expected_ja)
    
    def test_multiple_conditions_and_filter(self):
        """
        Test filtering with multiple conditions (AND operation).
        複数条件（AND演算）でのフィルタリングをテストします。
        """
        metadata = [
            {"category": "tech", "language": "ja", "difficulty": "beginner"},
            {"category": "legal", "language": "ja", "difficulty": "advanced"},
            {"category": "tech", "language": "en", "difficulty": "beginner"},
            {"category": "tech", "language": "ja", "difficulty": "advanced"},
        ]
        
        filter_engine = MetadataFilter(metadata)
        
        # Filter by category AND language
        # カテゴリと言語でフィルタリング
        tech_ja_docs = filter_engine.apply_filter({
            "category": "tech",
            "language": "ja"
        })
        expected = np.array([0, 3], dtype=np.int32)
        np.testing.assert_array_equal(tech_ja_docs, expected)
        
        # Filter by all three conditions
        # 3つの条件すべてでフィルタリング
        specific_docs = filter_engine.apply_filter({
            "category": "tech",
            "language": "ja",
            "difficulty": "beginner"
        })
        expected_specific = np.array([0], dtype=np.int32)
        np.testing.assert_array_equal(specific_docs, expected_specific)
    
    def test_multiple_values_or_filter(self):
        """
        Test filtering with multiple values for same field (OR operation).
        同じフィールドに対する複数値（OR演算）でのフィルタリングをテストします。
        """
        metadata = [
            {"category": "tech", "difficulty": "beginner"},
            {"category": "legal", "difficulty": "intermediate"},
            {"category": "medical", "difficulty": "advanced"},
            {"category": "tech", "difficulty": "advanced"},
        ]
        
        filter_engine = MetadataFilter(metadata)
        
        # Filter by multiple categories
        # 複数のカテゴリでフィルタリング
        multiple_categories = filter_engine.apply_filter({
            "category": ["tech", "medical"]
        })
        expected = np.array([0, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(multiple_categories, expected)
        
        # Filter by multiple difficulties
        # 複数の難易度でフィルタリング
        multiple_difficulties = filter_engine.apply_filter({
            "difficulty": ["beginner", "advanced"]
        })
        expected_diff = np.array([0, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(multiple_difficulties, expected_diff)
    
    def test_list_value_filtering(self):
        """
        Test filtering with list values in metadata.
        メタデータのリスト値でのフィルタリングをテストします。
        """
        metadata = [
            {"tags": ["python", "machine-learning"], "category": "tech"},
            {"tags": ["law", "contracts"], "category": "legal"},
            {"tags": ["python", "web-development"], "category": "tech"},
            {"tags": ["machine-learning", "deep-learning"], "category": "tech"},
        ]
        
        filter_engine = MetadataFilter(metadata)
        
        # Filter by tag
        # タグでフィルタリング
        python_docs = filter_engine.apply_filter({"tags": "python"})
        expected_python = np.array([0, 2], dtype=np.int32)
        np.testing.assert_array_equal(python_docs, expected_python)
        
        # Filter by another tag
        # 別のタグでフィルタリング
        ml_docs = filter_engine.apply_filter({"tags": "machine-learning"})
        expected_ml = np.array([0, 3], dtype=np.int32)
        np.testing.assert_array_equal(ml_docs, expected_ml)
    
    def test_empty_filter_returns_all(self):
        """
        Test that empty filter returns all documents.
        空のフィルタがすべての文書を返すことをテストします。
        """
        metadata = [
            {"category": "tech"},
            {"category": "legal"},
            {"category": "medical"},
        ]
        
        filter_engine = MetadataFilter(metadata)
        all_docs = filter_engine.apply_filter({})
        expected = np.array([0, 1, 2], dtype=np.int32)
        np.testing.assert_array_equal(all_docs, expected)
    
    def test_no_matches_returns_empty(self):
        """
        Test that filter with no matches returns empty array.
        マッチしないフィルタが空の配列を返すことをテストします。
        """
        metadata = [
            {"category": "tech"},
            {"category": "legal"},
        ]
        
        filter_engine = MetadataFilter(metadata)
        no_matches = filter_engine.apply_filter({"category": "nonexistent"})
        expected = np.array([], dtype=np.int32)
        np.testing.assert_array_equal(no_matches, expected)
    
    def test_weight_mask_creation(self):
        """
        Test weight mask creation from filtered indices.
        フィルタリングされたインデックスからのウェイトマスク作成をテストします。
        """
        metadata = [
            {"category": "tech"},
            {"category": "legal"},
            {"category": "tech"},
            {"category": "medical"},
        ]
        
        filter_engine = MetadataFilter(metadata)
        filtered_indices = np.array([0, 2], dtype=np.int32)
        weight_mask = filter_engine.create_weight_mask(filtered_indices, 4)
        
        expected_mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(weight_mask, expected_mask)


class TestMetadataValidation:
    """
    Test class for metadata validation functionality.
    メタデータ検証機能のテストクラスです。
    """
    
    def test_valid_metadata(self):
        """
        Test validation of valid metadata.
        有効なメタデータの検証をテストします。
        """
        valid_metadata = [
            {"category": "tech", "language": "ja"},
            {"category": "legal", "language": "en"},
        ]
        assert validate_metadata(valid_metadata) is True
    
    def test_invalid_metadata_not_list(self):
        """
        Test validation of invalid metadata (not a list).
        無効なメタデータ（リストでない）の検証をテストします。
        """
        invalid_metadata = {"category": "tech"}
        assert validate_metadata(invalid_metadata) is False
    
    def test_invalid_metadata_not_dict_items(self):
        """
        Test validation of invalid metadata (items not dictionaries).
        無効なメタデータ（項目が辞書でない）の検証をテストします。
        """
        invalid_metadata = ["not_a_dict", {"category": "tech"}]
        assert validate_metadata(invalid_metadata) is False
    
    def test_empty_metadata(self):
        """
        Test validation of empty metadata.
        空のメタデータの検証をテストします。
        """
        empty_metadata = []
        assert validate_metadata(empty_metadata) is True


class TestBM25MetadataIntegration:
    """
    Test class for BM25 integration with metadata filtering.
    メタデータフィルタリングとのBM25統合のテストクラスです。
    """
    
    def test_bm25_with_metadata_initialization(self):
        """
        Test BM25 initialization with metadata.
        メタデータでのBM25初期化をテストします。
        """
        metadata = [
            {"category": "tech", "language": "ja"},
            {"category": "legal", "language": "ja"},
            {"category": "medical", "language": "ja"},
        ]
        
        bm25 = BM25(metadata=metadata)
        assert bm25.metadata == metadata
        assert bm25.metadata_filter is not None
        assert isinstance(bm25.metadata_filter, MetadataFilter)
    
    def test_bm25_index_with_metadata(self):
        """
        Test BM25 indexing with metadata.
        メタデータでのBM25インデックス作成をテストします。
        """
        corpus = ["技術文書について", "法律文書について", "医療文書について"]
        metadata = [
            {"category": "tech", "language": "ja"},
            {"category": "legal", "language": "ja"},
            {"category": "medical", "language": "ja"},
        ]
        
        bm25 = BM25()
        bm25.index(corpus, metadata=metadata)
        
        assert bm25.metadata == metadata
        assert bm25.metadata_filter is not None
        assert len(bm25.metadata_filter.metadata) == 3
    
    def test_bm25_retrieve_with_filter(self):
        """
        Test BM25 retrieval with metadata filtering.
        メタデータフィルタリングでのBM25検索をテストします。
        """
        corpus = ["技術文書について", "法律文書について", "医療文書について", "技術資料について"]
        metadata = [
            {"category": "tech", "language": "ja"},
            {"category": "legal", "language": "ja"},
            {"category": "medical", "language": "ja"},
            {"category": "tech", "language": "ja"},
        ]
        
        bm25 = BM25()
        bm25.index(corpus, metadata=metadata)
        
        # Retrieve with filter
        # フィルタありで検索
        query = [["技術"]]
        results = bm25.retrieve(
            query,
            k=10,
            filter={"category": "tech"},
            return_metadata=True
        )
        
        # Should only return tech documents
        # 技術文書のみを返すべき
        assert len(results.documents[0]) == 2  # Two tech documents
        assert all(meta["category"] == "tech" for meta in results.metadata[0])
    
    def test_bm25_retrieve_no_matches(self):
        """
        Test BM25 retrieval with filter that matches no documents.
        文書にマッチしないフィルタでのBM25検索をテストします。
        """
        corpus = ["技術文書について", "法律文書について"]
        metadata = [
            {"category": "tech", "language": "ja"},
            {"category": "legal", "language": "ja"},
        ]
        
        bm25 = BM25()
        bm25.index(corpus, metadata=metadata)
        
        # Retrieve with filter that matches nothing
        # 何もマッチしないフィルタで検索
        query = [["技術"]]
        results = bm25.retrieve(
            query,
            k=10,
            filter={"category": "nonexistent"},
            return_metadata=True
        )
        
        # Should return empty results
        # 空の結果を返すべき
        assert len(results.documents[0]) == 0
        assert len(results.scores[0]) == 0
        assert results.metadata == []
    
    def test_bm25_retrieve_without_metadata_filter_error(self):
        """Test error handling when trying to filter without metadata / メタデータなしでフィルタリングしようとする際のエラーハンドリングテスト"""
        # Create BM25 instance without metadata
        # メタデータなしでBM25インスタンスを作成
        corpus = ["Python programming", "Web development"]
        bm25 = BM25()
        bm25.index(corpus)
        
        # Should raise error when trying to filter without metadata
        # メタデータなしでフィルタリングしようとするとエラーが発生するべき
        query = [["技術"]]
        with self.assertRaises(ValueError):
            bm25.retrieve(
                query,
                k=5,
                filter={"category": "tech"}
            )
    
    def test_bm25_invalid_metadata_error(self):
        """Test error handling for invalid metadata format / 無効なメタデータ形式のエラーハンドリングテスト"""
        # Test invalid metadata format (not all dict)
        # 無効なメタデータ形式をテスト（すべてdictではない）
        invalid_metadata = ["not_a_dict", {"category": "tech"}]
        
        with self.assertRaises(ValueError):
            BM25(metadata=invalid_metadata)


class TestAdvancedMetadataFiltering:
    """
    Test class for advanced metadata filtering features.
    高度なメタデータフィルタリング機能のテストクラスです。
    """
    
    def setup_method(self):
        """Setup test data for advanced filtering tests / 高度フィルタリングテスト用のテストデータを設定します"""
        self.corpus = [
            "Introduction to Python programming with data structures",
            "JavaScript fundamentals for web development", 
            "Advanced Java programming patterns",
            "HTML markup language basics",
            "CSS styling and design principles"
        ]
        
        self.metadata = [
            {
                "category": "tech", 
                "difficulty": "beginner", 
                "score": 0.95,
                "tags": ["python", "ai", "data-science"],
                "date": "2024-01-01",
                "language": "en"
            },
            {
                "category": "tech", 
                "difficulty": "intermediate", 
                "score": 0.85,
                "tags": ["javascript", "web", "frontend"],
                "date": "2024-01-15",
                "language": "en"  
            },
            {
                "category": "tech", 
                "difficulty": "advanced", 
                "score": 0.90,
                "tags": ["java", "enterprise", "backend"],
                "date": "2024-02-01",
                "language": "en"
            },
            {
                "category": "web", 
                "difficulty": "beginner", 
                "score": 0.75,
                "tags": ["html", "markup", "web"],
                "date": "2024-01-10",
                "language": "en"
            },
            {
                "category": "web", 
                "difficulty": "beginner", 
                "score": 0.80,
                "tags": ["css", "styling", "design"],
                "date": "2024-01-20",
                "language": "en"
            }
        ]
        
        # Create BM25 instance with metadata using proper tokenization
        # 適切なトークン化を使用してメタデータ付きBM25インスタンスを作成
        self.bm25 = BM25()
        corpus_tokens = bm25s.tokenize(self.corpus, stopwords="en", stemmer=None)
        self.bm25.index(corpus_tokens, metadata=self.metadata)
    
    def test_comparison_operators(self):
        """Test comparison operators / 比較演算子をテストします"""
        # Greater than or equal
        results = self.bm25.retrieve([["python"]], k=5, filter={"score": {"$gte": 0.85}})
        indices = results.documents[0]  # Retrieved document indices
        scores_match = all(self.metadata[idx]["score"] >= 0.85 for idx in indices)
        assert scores_match, "All returned documents should have score >= 0.85"
        
        # Less than
        results = self.bm25.retrieve([["python"]], k=5, filter={"score": {"$lt": 0.85}})
        indices = results.documents[0]
        scores_match = all(self.metadata[idx]["score"] < 0.85 for idx in indices)
        assert scores_match, "All returned documents should have score < 0.85"
        
        # Not equal
        results = self.bm25.retrieve([["python"]], k=5, filter={"category": {"$ne": "programming"}})
        indices = results.documents[0]
        categories_match = all(self.metadata[idx]["category"] != "programming" for idx in indices)
        assert categories_match, "All returned documents should not have category 'programming'"
        
        # Date comparison (string comparison)
        results = self.bm25.retrieve([["python"]], k=5, filter={"date": {"$gte": "2024-01-15"}})
        indices = results.documents[0]
        dates_match = all(self.metadata[idx]["date"] >= "2024-01-15" for idx in indices)
        assert dates_match, "All returned documents should have date >= '2024-01-15'"
    
    def test_in_operators(self):
        """Test $in and $nin operators / $inと$nin演算子をテストします"""
        # $in operator
        results = self.bm25.retrieve([["python"]], k=5, filter={"difficulty": {"$in": ["beginner", "advanced"]}})
        indices = results.documents[0]
        difficulties_match = all(self.metadata[idx]["difficulty"] in ["beginner", "advanced"] for idx in indices)
        assert difficulties_match, "All returned documents should have difficulty in ['beginner', 'advanced']"
        
        # $nin operator
        results = self.bm25.retrieve([["python"]], k=5, filter={"difficulty": {"$nin": ["intermediate"]}})
        indices = results.documents[0]
        difficulties_match = all(self.metadata[idx]["difficulty"] != "intermediate" for idx in indices)
        assert difficulties_match, "All returned documents should not have difficulty 'intermediate'"
    
    def test_exists_operator(self):
        """Test exists operator / 存在演算子をテストします"""
        # Test with a token that appears in multiple documents (programming)
        # 複数の文書に現れるトークン（programming）でテスト
        results = self.bm25.retrieve([["programming"]], k=5, filter={"category": {"$exists": True}})
        # At least some documents should be returned (those with actual scores > 0)
        # 少なくともいくつかの文書が返されるはず（実際のスコア > 0のもの）
        assert len(results.documents[0]) >= 1, "At least one document should have 'category' field and non-zero score"
        
        # Verify all returned documents have the category field
        # 返されたすべての文書がcategoryフィールドを持っていることを確認
        for doc_idx in results.documents[0]:
            assert "category" in self.metadata[doc_idx], f"Document {doc_idx} should have 'category' field"
        
        # Test non-existent field with $exists: False
        results = self.bm25.retrieve([["programming"]], k=5, filter={"author": {"$exists": False}})
        # Should return documents that don't have 'author' field (all of them in this case)
        # 'author'フィールドを持たない文書を返すはず（この場合はすべて）
        assert len(results.documents[0]) >= 1, "Should return documents without 'author' field"
        
        # Verify no returned documents have the author field  
        # 返された文書がauthorフィールドを持っていないことを確認
        for doc_idx in results.documents[0]:
            assert "author" not in self.metadata[doc_idx], f"Document {doc_idx} should not have 'author' field"
        
        # Test non-existent field with $exists: True (should return nothing with scores > 0)
        results = self.bm25.retrieve([["programming"]], k=5, filter={"author": {"$exists": True}})
        assert len(results.documents[0]) == 0, "No documents should have 'author' field"
    
    def test_regex_operator(self):
        """Test regular expression operator / 正規表現演算子をテストします"""
        # Find documents with tags starting with 'java'
        results = self.bm25.retrieve([["python"]], k=5, filter={"tags": {"$regex": "java.*"}})
        indices = results.documents[0]
        java_matches = []
        for idx in indices:
            doc_tags = self.metadata[idx]["tags"]
            has_java_tag = any("java" in tag.lower() for tag in doc_tags)
            java_matches.append(has_java_tag)
        assert all(java_matches), "All returned documents should have tags containing 'java'"
        
        # Find documents with tags containing 'web'
        results = self.bm25.retrieve([["python"]], k=5, filter={"tags": {"$regex": ".*web.*"}})
        indices = results.documents[0]
        web_matches = []
        for idx in indices:
            doc_tags = self.metadata[idx]["tags"]
            has_web_tag = any("web" in tag.lower() for tag in doc_tags)
            web_matches.append(has_web_tag)
        assert all(web_matches), "All returned documents should have tags containing 'web'"
    
    def test_logical_or_operator(self):
        """Test logical OR operator / 論理OR演算子をテストします"""
        results = self.bm25.retrieve([["python"]], k=5, filter={
            "$or": [
                {"category": "web-development"},
                {"difficulty": "advanced"}
            ]
        })
        
        indices = results.documents[0]
        or_matches = []
        for idx in indices:
            doc_meta = self.metadata[idx]
            matches_condition = (doc_meta["category"] == "web-development" or doc_meta["difficulty"] == "advanced")
            or_matches.append(matches_condition)
        assert all(or_matches), "All returned documents should match OR condition"
    
    def test_logical_and_operator(self):
        """Test logical AND operator / 論理AND演算子をテストします"""
        results = self.bm25.retrieve([["python"]], k=5, filter={
            "$and": [
                {"category": "programming"},
                {"difficulty": "beginner"}
            ]
        })
        
        indices = results.documents[0]
        and_matches = []
        for idx in indices:
            doc_meta = self.metadata[idx]
            matches_condition = (doc_meta["category"] == "programming" and doc_meta["difficulty"] == "beginner")
            and_matches.append(matches_condition)
        assert all(and_matches), "All returned documents should match AND condition"
    
    def test_logical_not_operator(self):
        """Test logical NOT operator / 論理NOT演算子をテストします"""
        results = self.bm25.retrieve([["python"]], k=5, filter={
            "$not": {"category": "programming"}
        })
        
        indices = results.documents[0]
        not_matches = []
        for idx in indices:
            doc_meta = self.metadata[idx]
            matches_condition = (doc_meta["category"] != "programming")  
            not_matches.append(matches_condition)
        assert all(not_matches), "All returned documents should match NOT condition"
    
    def test_complex_nested_conditions(self):
        """Test complex nested logical conditions / 複雑な入れ子論理条件をテストします"""
        # Complex condition: (category="programming" AND score >= 0.85) OR (category="web-development" AND difficulty="beginner")
        # 複雑な条件: (category="programming" AND score >= 0.85) OR (category="web-development" AND difficulty="beginner")
        results = self.bm25.retrieve([["python"]], k=5, filter={
            "$or": [
                {
                    "$and": [
                        {"category": "programming"},
                        {"score": {"$gte": 0.85}}
                    ]
                },
                {
                    "$and": [
                        {"category": "web-development"},
                        {"difficulty": "beginner"}
                    ]
                }
            ]
        })
        
        indices = results.documents[0]
        complex_matches = []
        for idx in indices:
            doc_meta = self.metadata[idx]
            condition1 = (doc_meta["category"] == "programming" and doc_meta["score"] >= 0.85)
            condition2 = (doc_meta["category"] == "web-development" and doc_meta["difficulty"] == "beginner")
            matches_condition = (condition1 or condition2)
            complex_matches.append(matches_condition)
        assert all(complex_matches), "All returned documents should match complex nested condition" 