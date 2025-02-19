# サンプルコード
import bm25s
import math

def main():
    # https://ja.wikipedia.org/wiki/%E6%9A%81%E7%BE%8E%E3%81%BB%E3%82%80%E3%82%89
    corpus = [
        "暁美 ほむら（あけみ ほむら）は、テレビアニメ『魔法少女まどか☆マギカ』に登場する架空の人物。まどか☆マギカの外伝漫画・『魔法少女おりこ☆マギカ』、『魔法少女まどか☆マギカ 〜The different story〜』『魔法少女まどか☆マギカ [魔獣編]』にも登場する。",
        "「時間操作」の魔法を操る魔法少女として設定されており、劇中では人間社会から持ち出した銃や爆弾の数々を時間操作能力と組み合わせて戦っている。",
        "劇中で直接そのように呼ばれる場面はないが[2][注 1]、ファンからは「ほむほむ」という愛称で呼ばれている[2][5]。",
        "一人称は「私」。",
        "まどかは「まどか」、さやかは「美樹さやか」、マミは「巴マミ」、杏子は「杏子」と呼び、まどかと杏子以外の魔法少女はフルネームで呼び捨てにしている。",
        "声優は各作品共通で斎藤千和（英語版はクリスティーナ・ヴィー）が担当する。『マギアレコード 魔法少女まどか☆マギカ外伝』の舞台版では河田陽菜（けやき坂46（現・日向坂46））が演じる[6]。",
    ]
    
    corpus_tokens = bm25s.tokenize(corpus, stopwords="japanese")
    print(corpus_tokens)

    retriever = bm25s.BM25(use_log_normalization=False)
    retriever.index(corpus_tokens)

    query = "ほむらは誰？"
    query_tokens = bm25s.tokenize(query, stopwords="japanese")
    print(query_tokens)

    results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")


    # 保存
    retriever.save("retriever_magical_girl.pkl")

    # ロード
    retriever = bm25s.BM25.load("retriever_magical_girl.pkl")

    # ロード後に変更ないのを確認
    query = "ほむらは誰？"
    query_tokens = bm25s.tokenize(query, stopwords="japanese")
    print(query_tokens)

    results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)
    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")


if __name__ == "__main__":
    main()
