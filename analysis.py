import nltk
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba  # Library for Chinese word segmentation

# References:
# 1. https://investigate.ai/text-analysis/using-tf-idf-with-chinese/
# 2. https://github.com/topics/chinese-sentiment-analysis?l=python
# 3. https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/Multilingual/Chinese/03-POS-Keywords-Chinese.html
# 5. https://okan.cloud/posts/2022-01-16-text-vectorization-using-python-tf-idf/
# 6. https://medium.com/nlplanet/text-analysis-topic-modelling-with-spacy-gensim-4cd92ef06e06
# nltk.download()

def review_analysis(df,col):
    # 1. Lower case letters
    # 2. n-grams
    # 3. stemming
    # 4. stop words
    # 5. TF-IDF
    chinese_stopwords = nltk.corpus.stopwords.words('chinese')
    lemma = WordNetLemmatizer()

    length = df.shape[0]
    data_without_stopwords = []

    # Loop through each reviews
    for i in range(0, length):
        reviews = df.iloc[i][col] # extract reviews
        doc = jieba.lcut(reviews.strip()) # Split the Chinese words

        doc = [lemma.lemmatize(word) for word in doc if not word in set(chinese_stopwords)]
        # remove characterists
        special_symbols = ['，','！','。','？','h','+', ' ', '、', '?','…','：',')','⊙','o','⊙','(',
                 '!',':','','...', "'"]
        doc = [value for value in doc if not value in set(special_symbols)]
        if (len(doc) > 0):
            data_without_stopwords.append(doc)

    #
    data_without_stopwords = [item for t in data_without_stopwords for item in t]
    data_without_stopwords = remove_duplicates(data_without_stopwords)

    # Import Tfidf vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data_without_stopwords)
    print("n_samples: %d, n_features: %d" % vectors.shape)

    # Select the first five documents from the data set
    n = 1000
    tf_idf = pd.DataFrame(vectors.todense()).iloc[:n]
    tf_idf.columns = vectorizer.get_feature_names()
    tf_idf.to_csv('test.csv')
    tfidf_matrix = tf_idf.T
    tfidf_matrix.columns = ['response' + str(i) for i in range(1, n )]
    tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)


    # Top 10 words
    tfidf_matrix = tfidf_matrix.sort_values(by='count', ascending=False)[:100]

    # Print the first 10 words
    print(tfidf_matrix.drop(columns=['count']).head(10))

def english_word_removal(doc):
    doc = re.sub('[^a-zA-Z]', ' ', doc)
    doc = doc.lower()
    doc = doc.split()
    print(doc)


def remove_duplicates(words_list):
    temp = []
    for x in words_list:
        if x not in temp:
            temp.append(x)

    return temp

def TF_IDF():
    return None
