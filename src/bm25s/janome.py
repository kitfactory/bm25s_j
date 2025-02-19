from typing import List, Union, Callable
from janome.tokenizer import Tokenizer as JanomeTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

def tokenize(
    texts: Union[str, List[str]],
    lower: bool = True,
    stopwords: Union[str, List[str]] = None,
    pos_filter: List[str] = ["名詞", "動詞", "形容詞"],  # 品詞フィルター（デフォルトは主要な品詞のみ）
    show_progress: bool = True,
    leave_progress: bool = False,
    allow_empty: bool = True,
) -> "Tokenized":
    """
    Tokenize Japanese text using Janome tokenizer.
    日本語テキストをJanomeトークナイザーを使用してトークン化します。

    Parameters
    ----------
    texts : Union[str, List[str]]
        Text or list of texts to tokenize
        トークン化するテキストまたはテキストのリスト
    
    lower : bool, optional
        Whether to convert text to lowercase
        テキストを小文字に変換するかどうか
    
    stopwords : Union[str, List[str]], optional
        Stopwords to remove. Can be a list of words or a string specifying a predefined stopword list
        除去するストップワード。単語のリストまたは事前定義されたストップワードリストを指定する文字列
    
    pos_filter : List[str], optional
        Part-of-speech tags to keep. Default is ["名詞", "動詞", "形容詞"]
        保持する品詞タグ。デフォルトは ["名詞", "動詞", "形容詞"]
    
    show_progress : bool, optional
        Whether to show progress bar
        進捗バーを表示するかどうか
    
    leave_progress : bool, optional
        Whether to leave progress bar after completion
        完了後に進捗バーを残すかどうか
    
    allow_empty : bool, optional
        Whether to allow empty token lists
        空のトークンリストを許可するかどうか

    Returns
    -------
    Tokenized
        A named tuple containing token IDs and vocabulary
        トークンIDと語彙を含む名前付きタプル
    """
    if isinstance(texts, str):
        texts = [texts]

    # Lazy import of _infer_stopwords and Tokenized to avoid circular dependency
    from .tokenization import _infer_stopwords, Tokenized

    # Initialize Janome tokenizer
    tokenizer = JanomeTokenizer()
    
    # Process stopwords using the locally imported _infer_stopwords
    stopwords_set = set(_infer_stopwords(stopwords)) if stopwords else set()

    # Initialize vocabulary dictionary
    vocab_dict = {}
    corpus_ids = []

    # Process each text
    for text in tqdm(
        texts,
        desc="Tokenizing texts",
        disable=not show_progress,
        leave=leave_progress
    ):
        if lower:
            text = text.lower()

        # Tokenize and filter by POS
        tokens = []
        for token in tokenizer.tokenize(text):
            # Check if the token's POS starts with any of the allowed POS tags
            if any(token.part_of_speech.startswith(pos) for pos in pos_filter):
                surface = token.surface
                if surface not in stopwords_set:
                    tokens.append(surface)

        # Handle empty documents if allowed
        if len(tokens) == 0 and allow_empty:
            if "" not in vocab_dict:
                vocab_dict[""] = len(vocab_dict)
            tokens = [""]

        # Convert tokens to IDs
        doc_ids = []
        for token in tokens:
            if token not in vocab_dict:
                vocab_dict[token] = len(vocab_dict)
            doc_ids.append(vocab_dict[token])

        corpus_ids.append(doc_ids)

    return Tokenized(ids=corpus_ids, vocab=vocab_dict)
