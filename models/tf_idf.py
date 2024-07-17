import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def TF_IDF(words_list, n=5):
    # Covert list of list to list of string
    words_list = [item for t in words_list for item in t]
    words_list = remove_duplicates(words_list)

    # Import Tfidf vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(words_list)
    print("n_samples: %d, n_features: %d" % vectors.shape)

    # Select the first n documents from the data set
    tf_idf = pd.DataFrame(vectors.todense()).iloc[:n]
    tf_idf.columns = vectorizer.get_feature_names()

    tfidf_matrix = tf_idf.T
    tfidf_matrix.columns = ['response' + str(i) for i in range(1, n)]
    tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)

    # Top 50 words
    tfidf_matrix = tfidf_matrix.sort_values(by='count', ascending=False)[:50]

    # Print the first 10 words
    print(tfidf_matrix.drop(columns=['count']).head(10))


def remove_duplicates(words_list):
    temp = []
    for x in words_list:
        if x not in temp:
            temp.append(x)

    return temp
