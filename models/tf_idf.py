import pandas as pd
import tensorflow as tf

def TF_IDF(words_list, n=5, max_tokens=20000):
    if not words_list:
        return pd.DataFrame()

    documents = [" ".join(tokens) for tokens in words_list if tokens]
    if not documents:
        return pd.DataFrame()

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        max_tokens=max_tokens,
        output_mode="tf_idf",
    )
    vectorizer.adapt(documents)

    matrix = vectorizer(tf.constant(documents)).numpy()
    vocabulary = vectorizer.get_vocabulary()
    tf_idf = pd.DataFrame(matrix, columns=vocabulary).iloc[:n].copy()

    tfidf_matrix = tf_idf.T
    tfidf_matrix.columns = [f"response{i}" for i in range(1, tfidf_matrix.shape[1] + 1)]
    tfidf_matrix["count"] = tfidf_matrix.sum(axis=1)

    tfidf_matrix = tfidf_matrix.sort_values(by="count", ascending=False).head(50)
    print(tfidf_matrix.drop(columns=["count"]).head(10))
    return tfidf_matrix


def remove_duplicates(words_list):
    return list(dict.fromkeys(words_list))
