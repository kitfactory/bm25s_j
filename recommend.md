# bm25s-j: bm25sの日本語対応版作った

本記事では、[bm25s-j](https://github.com/kitfactory/bm25s_j) の紹介をします。  

本家の[bm25s](https://github.com/xhluca/bm25s) を日本語対応に改良したもので、PyPI に登録済みのため、インストールも簡単です。

## 1. bm25s の特徴
PythonでBM25アルゴリズムの高速、低依存、低メモリ実装を提供するように設計されています。
Python で最も人気のある BM25 実装である rank-bm25 (シングルスレッド設定) を基準にした、 Elastic との クエリ数の比は以下のようになるそうです。

__bm25sのベンチマーク結果:本家画像__

![benchmark](https://bm25s.github.io/assets/comparison.png)

## 2. bm25s-jでの対応

### 2.1. 日本語形態素解析

日本語の文章も適切にトークン化できるよう、`janome` を利用しています。これにより、英語だけでなく日本語文書に対しても高い検索精度を実現しています。また、日本語のストップワードを取り込んでいます。

### 2.2. Windows環境対応

従来の bm25s は、メモリ使用量取得に Unix の `resource` モジュールに依存していましたが、Windows環境ではメモリ使用量を正確に取得できるよう、`psutil` パッケージを利用しています。

## 3. サンプル

### 3.1. インストール

インストールはpipで行う事が可能です。

```bash
pip install bm25s-j
```

### 3.2. サンプルコード

以下は、魔法少女まどか☆マギカに登場する暁美ほむらに関するWikipediaの情報をコーパスを使い、bm25s-j で検索を行うサンプルコードです。

```python
import bm25s

# ほむらの情報を含むコーパス例
corpus = [
    "暁美 ほむら（あけみ ほむら）は、テレビアニメ『魔法少女まどか☆マギカ』に登場する架空の人物です。",
    "ほむらはタイムリープを繰り返し、悲劇的な過去を持つ魔法少女です。",
    "物語の中で、ほむらは大切な使命を帯び、運命に立ち向かいます。"
]

# 日本語用のトークナイザーにより、コーパスをトークン化（stopwordsに"japanese"を指定）
corpus_tokens = bm25s.tokenize(corpus, stopwords="japanese")
print("コーパスのトークン:", corpus_tokens)

# BM25 インスタンスを作成し、コーパスのトークン化結果からインデックスを構築
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# クエリ例：「ほむらは誰？」
query = "ほむらは誰？"
query_tokens = bm25s.tokenize(query, stopwords="japanese")
print("クエリのトークン:", query_tokens)

# 検索を実行し、上位2件の結果を取得
results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")
```



## 参考記事

本プロジェクトの発端・参考となったのは、  
[kun432 さんのZenn記事](https://zenn.dev/kun432/scraps/5ae46c49a92bcb)です。  
この記事をきっかけに、bm25s の素晴らしさに触発され、日本語対応を強化した bm25s-j が生まれました。

## 将来？

現在はインストールの簡便さからjanomeを使用しました。ただ、bm25sの高速さを遺憾なく発揮できてはいないと思うので、そのあたりかな。。。

## まとめ
[bm25s-j](https://github.com/kitfactory/bm25s_j) は、BM25 による優れた検索アルゴリズムと、簡単なインターフェースを提供します。
また、PyPI に登録済みで誰でも簡単にインストール可能です。