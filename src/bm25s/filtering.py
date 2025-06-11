"""
BM25S Metadata Filtering Module

This module provides metadata-based filtering functionality for BM25 retrieval.
BM25検索のためのメタデータベースフィルタリング機能を提供するモジュールです。
"""

from typing import Any, Dict, List, Optional, Union, Set
import numpy as np
import logging

logger = logging.getLogger("bm25s.filtering")


class MetadataFilter:
    """
    Metadata filtering engine for BM25 documents.
    BM25文書のためのメタデータフィルタリングエンジンです。
    """
    
    def __init__(self, metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize metadata filter with document metadata.
        文書メタデータでメタデータフィルターを初期化します。
        
        Parameters
        ----------
        metadata : List[Dict[str, Any]], optional
            List of metadata dictionaries, one for each document.
            各文書に対するメタデータ辞書のリストです。
        """
        self.metadata = metadata or []
        self._build_indices()
    
    def _build_indices(self):
        """
        Build internal indices for efficient filtering.
        効率的なフィルタリングのための内部インデックスを構築します。
        """
        self.field_indices: Dict[str, Dict[Any, Set[int]]] = {}
        
        for doc_idx, doc_metadata in enumerate(self.metadata):
            if not isinstance(doc_metadata, dict):
                continue
                
            for field, value in doc_metadata.items():
                if field not in self.field_indices:
                    self.field_indices[field] = {}
                
                # Handle list values
                # リスト値を処理します
                if isinstance(value, list):
                    for item in value:
                        if item not in self.field_indices[field]:
                            self.field_indices[field][item] = set()
                        self.field_indices[field][item].add(doc_idx)
                else:
                    if value not in self.field_indices[field]:
                        self.field_indices[field][value] = set()
                    self.field_indices[field][value].add(doc_idx)
    
    def apply_filter(self, filter_conditions: Dict[str, Any]) -> np.ndarray:
        """
        Apply filter conditions and return document indices that match.
        フィルタ条件を適用し、マッチする文書インデックスを返します。
        
        Supports logical operators: $or, $and, $not
        論理演算子をサポート: $or, $and, $not
        
        Parameters
        ----------
        filter_conditions : Dict[str, Any]
            Dictionary containing filter conditions.
            フィルタ条件を含む辞書です。
        
        Returns
        -------
        np.ndarray
            Array of document indices that match the filter conditions.
            フィルタ条件にマッチする文書インデックスの配列です。
        """
        if not filter_conditions:
            return np.arange(len(self.metadata), dtype=np.int32)
        
        matching_docs = self._apply_logical_filter(filter_conditions)
        return np.array(sorted(matching_docs), dtype=np.int32)
    
    def _apply_logical_filter(self, filter_conditions: Dict[str, Any]) -> Set[int]:
        """
        Apply logical filter conditions with support for $or, $and, $not operators.
        $or, $and, $not演算子をサポートする論理フィルタ条件を適用します。
        
        Parameters
        ----------
        filter_conditions : Dict[str, Any]
            Dictionary containing filter conditions.
            フィルタ条件を含む辞書です。
        
        Returns
        -------
        Set[int]
            Set of document indices that match the filter conditions.
            フィルタ条件にマッチする文書インデックスの集合です。
        """
        # Handle logical operators
        # 論理演算子を処理
        if "$or" in filter_conditions:
            return self._apply_or_filter(filter_conditions["$or"])
        
        if "$and" in filter_conditions:
            return self._apply_and_filter(filter_conditions["$and"])
        
        if "$not" in filter_conditions:
            return self._apply_not_filter(filter_conditions["$not"])
        
        # Handle regular field conditions (implicit AND)
        # 通常のフィールド条件を処理（暗黙のAND）
        matching_docs = None
        
        for field, condition in filter_conditions.items():
            if field.startswith("$"):
                logger.warning(f"Unsupported logical operator: {field}")
                continue
                
            field_matches = self._apply_field_filter(field, condition)
            
            if matching_docs is None:
                matching_docs = field_matches
            else:
                # AND operation: intersection of matching documents
                # AND演算：マッチする文書の積集合
                matching_docs = matching_docs.intersection(field_matches)
        
        return matching_docs or set()
    
    def _apply_or_filter(self, conditions: List[Dict[str, Any]]) -> Set[int]:
        """
        Apply OR filter conditions.
        ORフィルタ条件を適用します。
        
        Parameters
        ----------
        conditions : List[Dict[str, Any]]
            List of filter condition dictionaries.
            フィルタ条件辞書のリストです。
        
        Returns
        -------
        Set[int]
            Set of document indices that match any of the conditions.
            いずれかの条件にマッチする文書インデックスの集合です。
        """
        matching_docs = set()
        
        for condition in conditions:
            condition_matches = self._apply_logical_filter(condition)
            matching_docs = matching_docs.union(condition_matches)
        
        return matching_docs
    
    def _apply_and_filter(self, conditions: List[Dict[str, Any]]) -> Set[int]:
        """
        Apply AND filter conditions.
        ANDフィルタ条件を適用します。
        
        Parameters
        ----------
        conditions : List[Dict[str, Any]]
            List of filter condition dictionaries.
            フィルタ条件辞書のリストです。
        
        Returns
        -------
        Set[int]
            Set of document indices that match all of the conditions.
            すべての条件にマッチする文書インデックスの集合です。
        """
        matching_docs = None
        
        for condition in conditions:
            condition_matches = self._apply_logical_filter(condition)
            
            if matching_docs is None:
                matching_docs = condition_matches
            else:
                matching_docs = matching_docs.intersection(condition_matches)
        
        return matching_docs or set()
    
    def _apply_not_filter(self, condition: Dict[str, Any]) -> Set[int]:
        """
        Apply NOT filter condition.
        NOTフィルタ条件を適用します。
        
        Parameters
        ----------
        condition : Dict[str, Any]
            Filter condition dictionary to negate.
            否定するフィルタ条件辞書です。
        
        Returns
        -------
        Set[int]
            Set of document indices that do not match the condition.
            条件にマッチしない文書インデックスの集合です。
        """
        all_docs = set(range(len(self.metadata)))
        matching_docs = self._apply_logical_filter(condition)
        return all_docs - matching_docs
    
    def _apply_field_filter(self, field: str, condition: Any) -> Set[int]:
        """
        Apply filter condition for a specific field.
        特定のフィールドに対するフィルタ条件を適用します。
        
        Parameters
        ----------
        field : str
            Field name to filter on.
            フィルタリング対象のフィールド名です。
        condition : Any
            Filter condition (value, list of values, or dict with operators).
            フィルタ条件（値、値のリスト、または演算子を含む辞書）です。
        
        Returns
        -------
        Set[int]
            Set of document indices that match the condition.
            条件にマッチする文書インデックスの集合です。
        """
        # Handle advanced operators first (including $exists) even if field doesn't exist
        # フィールドが存在しない場合でも高度な演算子（$existsを含む）を最初に処理
        if isinstance(condition, dict):
            return self._apply_operator_filter(field, condition)
        
        # For non-operator conditions, field must exist
        # 演算子以外の条件では、フィールドが存在する必要がある
        if field not in self.field_indices:
            return set()
        
        field_index = self.field_indices[field]
        
        # Simple equality filter
        # 単純な等価フィルタ
        if isinstance(condition, (str, int, float, bool)):
            return field_index.get(condition, set())
        
        # Multiple values filter (OR operation)
        # 複数値フィルタ（OR演算）
        elif isinstance(condition, list):
            matching_docs = set()
            for value in condition:
                matching_docs.update(field_index.get(value, set()))
            return matching_docs
        
        return set()
    
    def _apply_operator_filter(self, field: str, condition: Dict[str, Any]) -> Set[int]:
        """
        Apply advanced operator-based filter conditions.
        高度な演算子ベースのフィルタ条件を適用します。
        
        Supported operators:
        サポートされる演算子:
        - $gt, $gte, $lt, $lte: Numerical/string comparison
        - $ne: Not equal
        - $in, $nin: Value in/not in list
        - $exists: Field exists check
        - $regex: Regular expression matching
        
        Parameters
        ----------
        field : str
            Field name to filter on.
            フィルタリング対象のフィールド名です。
        condition : Dict[str, Any]
            Dictionary containing operator-based conditions.
            演算子ベースの条件を含む辞書です。
        
        Returns
        -------
        Set[int]
            Set of document indices that match the condition.
            条件にマッチする文書インデックスの集合です。
        """
        import re
        from typing import Union
        
        if field not in self.field_indices:
            # Field doesn't exist, handle $exists operator
            # フィールドが存在しない場合、$existsオペレータを処理
            if "$exists" in condition:
                if condition["$exists"]:
                    # Field should exist but doesn't - return empty set
                    # フィールドが存在すべきだが存在しない - 空集合を返す
                    return set()
                else:
                    # Field should not exist and doesn't - return all documents
                    # フィールドが存在すべきでなく実際に存在しない - すべての文書を返す
                    return set(range(len(self.metadata)))
            # For other operators, if field doesn't exist, no matches
            # 他の演算子の場合、フィールドが存在しなければマッチなし
            return set()
        
        field_index = self.field_indices[field]
        matching_docs = set()
        
        for operator, value in condition.items():
            if operator == "$gt":
                # Greater than
                # より大きい
                for field_value, doc_indices in field_index.items():
                    try:
                        if self._compare_values(field_value, value) > 0:
                            matching_docs.update(doc_indices)
                    except (TypeError, ValueError):
                        continue
                        
            elif operator == "$gte":
                # Greater than or equal
                # 以上
                for field_value, doc_indices in field_index.items():
                    try:
                        if self._compare_values(field_value, value) >= 0:
                            matching_docs.update(doc_indices)
                    except (TypeError, ValueError):
                        continue
                        
            elif operator == "$lt":
                # Less than
                # より小さい
                for field_value, doc_indices in field_index.items():
                    try:
                        if self._compare_values(field_value, value) < 0:
                            matching_docs.update(doc_indices)
                    except (TypeError, ValueError):
                        continue
                        
            elif operator == "$lte":
                # Less than or equal
                # 以下
                for field_value, doc_indices in field_index.items():
                    try:
                        if self._compare_values(field_value, value) <= 0:
                            matching_docs.update(doc_indices)
                    except (TypeError, ValueError):
                        continue
                        
            elif operator == "$ne":
                # Not equal
                # 等しくない
                for field_value, doc_indices in field_index.items():
                    if field_value != value:
                        matching_docs.update(doc_indices)
                        
            elif operator == "$in":
                # Value in list
                # 値がリストに含まれる
                if isinstance(value, list):
                    for v in value:
                        if v in field_index:
                            matching_docs.update(field_index[v])
                            
            elif operator == "$nin":
                # Value not in list
                # 値がリストに含まれない
                if isinstance(value, list):
                    excluded_values = set(value)
                    for field_value, doc_indices in field_index.items():
                        if field_value not in excluded_values:
                            matching_docs.update(doc_indices)
                            
            elif operator == "$exists":
                # Field exists
                # フィールドが存在する
                if value:
                    # Return all documents that have this field
                    # このフィールドを持つすべての文書を返す
                    for doc_indices in field_index.values():
                        matching_docs.update(doc_indices)
                else:
                    # Return documents that don't have this field
                    # このフィールドを持たない文書を返す
                    all_docs_with_field = set()
                    for doc_indices in field_index.values():
                        all_docs_with_field.update(doc_indices)
                    matching_docs.update(set(range(len(self.metadata))) - all_docs_with_field)
                    
            elif operator == "$regex":
                # Regular expression matching
                # 正規表現マッチング
                try:
                    pattern = re.compile(str(value))
                    for field_value, doc_indices in field_index.items():
                        if pattern.search(str(field_value)):
                            matching_docs.update(doc_indices)
                except re.error:
                    logger.warning(f"Invalid regex pattern: {value}")
                    
            else:
                logger.warning(f"Unsupported operator: {operator}")
        
        return matching_docs
    
    def _compare_values(self, value1: Any, value2: Any) -> int:
        """
        Compare two values for ordering.
        2つの値を順序比較します。
        
        Returns -1 if value1 < value2, 0 if equal, 1 if value1 > value2
        value1 < value2の場合-1、等しい場合0、value1 > value2の場合1を返します
        """
        # Try numeric comparison first
        # 最初に数値比較を試す
        try:
            num1 = float(value1)
            num2 = float(value2)
            if num1 < num2:
                return -1
            elif num1 > num2:
                return 1
            else:
                return 0
        except (ValueError, TypeError):
            pass
        
        # Fall back to string comparison
        # 文字列比較にフォールバック
        str1 = str(value1)
        str2 = str(value2)
        if str1 < str2:
            return -1
        elif str1 > str2:
            return 1
        else:
            return 0
    
    def create_weight_mask(self, filtered_indices: np.ndarray, total_docs: int) -> np.ndarray:
        """
        Create a weight mask for filtering during BM25 scoring.
        BM25スコアリング時のフィルタリング用ウェイトマスクを作成します。
        
        Parameters
        ----------
        filtered_indices : np.ndarray
            Array of document indices that should be included.
            含めるべき文書インデックスの配列です。
        total_docs : int
            Total number of documents.
            文書の総数です。
        
        Returns
        -------
        np.ndarray
            Binary weight mask where 1.0 indicates included documents.
            含まれる文書を1.0で示すバイナリウェイトマスクです。
        """
        weight_mask = np.zeros(total_docs, dtype=np.float32)
        weight_mask[filtered_indices] = 1.0
        return weight_mask


def validate_metadata(metadata: List[Dict[str, Any]]) -> bool:
    """
    Validate metadata format and structure.
    メタデータの形式と構造を検証します。
    
    Parameters
    ----------
    metadata : List[Dict[str, Any]]
        List of metadata dictionaries to validate.
        検証するメタデータ辞書のリストです。
    
    Returns
    -------
    bool
        True if metadata is valid, False otherwise.
        メタデータが有効な場合True、そうでなければFalseです。
    """
    if not isinstance(metadata, list):
        logger.error("Metadata must be a list")
        return False
    
    for i, item in enumerate(metadata):
        if not isinstance(item, dict):
            logger.error(f"Metadata item at index {i} must be a dictionary")
            return False
    
    return True 