# サンプルコード

import bm25s
import Stemmer

# コーパスの定義
corpus = [
    "cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]


# コーパスをトークン化してIDだけを残す（高速化、メモリ節約）
corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
print(corpus_tokens)

retriever = bm25s.BM25()
retriever.index(corpus_tokens)

query = "does the fish purr like a cat?"
query_tokens = bm25s.tokenize(query)

print(query_tokens)

# Tokenized(ids=[[0, 1, 4, 3, 2]], vocab={'doe': 0, 'fish': 1, 'cat': 2, 'like': 3, 'purr': 4})

# (doc id, score)のタプルとして、top-kの結果を得る。どちらも (n_queries, k) 形式の配列である。
results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)

print(results)
print(scores)


from bm25s.janome import tokenize

def main():
    # サンプルテキスト
    texts = [
        "これは日本語の文章です。",
        "形態素解析を使って文章を分かち書きします。",
        "BM25による検索のために、文章をトークン化します。"
    ]

    # 基本的な使用方法
    result = tokenize(texts)
    print("\n基本的なトークン化:")
    for i, (text, tokens) in enumerate(zip(texts, result.ids)):
        print(f"\n元の文章 {i+1}: {text}")
        # トークンIDを語彙に変換して表示
        tokens_str = [list(result.vocab.keys())[list(result.vocab.values()).index(tid)] for tid in tokens]
        print(f"トークン: {tokens_str}")

    # 品詞フィルターのカスタマイズ（名詞のみ）
    result_nouns = tokenize(texts, pos_filter=["名詞"])
    print("\n\n名詞のみのトークン化:")
    for i, (text, tokens) in enumerate(zip(texts, result_nouns.ids)):
        print(f"\n元の文章 {i+1}: {text}")
        tokens_str = [list(result_nouns.vocab.keys())[list(result_nouns.vocab.values()).index(tid)] for tid in tokens]
        print(f"トークン: {tokens_str}")

if __name__ == "__main__":
    main()



# retrieverの保存、ロード

