from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tf_idf(words):
    tfidf_vectorizer = TfidfVectorizer()
    vectors = tfidf_vectorizer.fit_transform(words)

    tf_idf = pd.DataFrame(vectors.todense()).iloc[:5]
    tf_idf.columns = tfidf_vectorizer.get_feature_names()
    tfidf_matrix = tf_idf.T
    tfidf_matrix.columns = ['response' + str(i) for i in range(1,6)]
    tfidf_matrix['count'] = tfidf_matrix.sum(axis = 1)

    # Top 10 words
    tfidf_matrix = tfidf_matrix.sort_values(by = 'count', ascending= False)[:10]
    print(tfidf_matrix.drop(columns = ['count']).head(10))
    return tfidf_matrix