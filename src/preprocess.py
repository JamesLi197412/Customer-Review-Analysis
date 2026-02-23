import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

try:
    import jieba
except ModuleNotFoundError:
    jieba = None

DEFAULT_STOPWORDS = {
    "",
    " ",
    "\n",
    "\t",
    "，",
    "。",
    "！",
    "？",
    "、",
    "；",
    "：",
    "（",
    "）",
    "(",
    ")",
    "...",
    "…",
    "'",
    '"',
    "+",
    "的",
    "了",
    "和",
    "是",
    "也",
    "都",
    "很",
    "就",
    "在",
}

SYMBOL_PATTERN = re.compile(r"^[\W_]+$")
FALLBACK_TOKENIZER_WARNED = False


def load_stopwords(stopwords_path=None):
    stopwords = set(DEFAULT_STOPWORDS)
    if not stopwords_path:
        return stopwords

    path = Path(stopwords_path)
    if not path.exists():
        return stopwords

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if value:
                stopwords.add(value)
    return stopwords


def normalize_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = str(text)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize_chinese_text(text, stopwords):
    global FALLBACK_TOKENIZER_WARNED

    if jieba is not None:
        raw_tokens = jieba.lcut(text)
    else:
        # Fallback tokenizer when jieba is unavailable:
        # keep CJK characters as single tokens and keep alphanumeric chunks.
        raw_tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)
        if not FALLBACK_TOKENIZER_WARNED:
            print("Warning: `jieba` is not installed. Using fallback tokenizer; install `jieba` for better tokenization.")
            FALLBACK_TOKENIZER_WARNED = True

    tokens = []
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        lower_token = token.lower()
        if lower_token in stopwords or token in stopwords:
            continue
        if SYMBOL_PATTERN.fullmatch(token):
            continue
        tokens.append(lower_token)
    return tokens


def data_preprocess(df, col, stopwords_path=None):
    if col not in df.columns:
        raise KeyError(f"Column '{col}' is missing from input dataframe.")

    stopwords = load_stopwords(stopwords_path)
    processed_df = df.copy()
    tokenized_reviews = []
    normalized_reviews = []

    for review in processed_df[col].fillna(""):
        normalized = normalize_text(review)
        tokens = tokenize_chinese_text(normalized, stopwords)
        normalized_reviews.append(" ".join(tokens))
        tokenized_reviews.append(tokens)

    processed_df["cleaned reviews"] = tokenized_reviews
    processed_df["normalized_text"] = normalized_reviews
    processed_df["token_count"] = [len(tokens) for tokens in tokenized_reviews]

    return processed_df, tokenized_reviews


def word_count(word_lists, output_path="output/analysis/word_frequency.txt"):
    results = map_reduce(word_lists)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for key, value in sorted(results.items(), key=lambda item: item[1], reverse=True):
            file.write(f"{key}:{value}\n")


def mapper(words):
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] += 1
    return word_freq


def reducer(word_freq_list):
    final_word_freq = defaultdict(int)
    for word_freq in word_freq_list:
        for word, freq in word_freq.items():
            final_word_freq[word] += freq
    return final_word_freq


def map_reduce(words_list):
    if not words_list:
        return defaultdict(int)

    process_count = min(4, max(1, cpu_count() - 1))
    with Pool(processes=process_count) as pool:
        mapped = pool.map(mapper, words_list)
    return reducer(mapped)


def english_word_removal(doc):
    doc = re.sub("[^a-zA-Z]", " ", doc)
    doc = doc.lower()
    return doc.split()


def remove_duplicates(words_list):
    return list(dict.fromkeys(words_list))
