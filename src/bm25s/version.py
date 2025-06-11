#!/usr/bin/env python3
"""
Version information for BM25s-j package.
BM25s-jパッケージのバージョン情報です。
"""

# Version of the BM25s-j package
# BM25s-jパッケージのバージョン
__version__ = "0.2.0"

# Version info as a tuple for programmatic access  
# プログラムからアクセス用のタプル形式バージョン情報
VERSION_INFO = (0, 2, 0)

# Release information
# リリース情報
RELEASE_DATE = "2025-06-11"
RELEASE_NOTES = """
Release 0.2.0: Advanced Metadata Filtering
リリース 0.2.0: 高度なメタデータフィルタリング

New Features / 新機能:
- ✅ Advanced metadata filtering with comparison operators ($gt, $gte, $lt, $lte, $ne)
- ✅ List operators ($in, $nin) for value inclusion/exclusion
- ✅ Existence operator ($exists) for field presence checking
- ✅ Regular expression operator ($regex) for pattern matching
- ✅ Logical operators ($or, $and, $not) for complex conditions
- ✅ Nested logical conditions support
- ✅ Comprehensive test suite for all filtering operations
- ✅ Integration with BM25 scoring system
- ✅ Performance optimizations for filtered search

Improvements / 改善:
- 🔧 Fixed top-k selection bounds error with dynamic k adjustment
- 🔧 Resolved weight mask integration with BM25 scoring
- 🔧 Enhanced numba backend support for filtering
- 🔧 Improved error handling and user feedback
- 🔧 Better memory efficiency with early document filtering

Breaking Changes / 破壊的変更:
- None for this release / このリリースではなし

Bug Fixes / バグ修正:
- 🐛 Fixed $exists: False operator for non-existent fields
- 🐛 Resolved zero-score filtering issue
- 🐛 Fixed empty result handling
"""
