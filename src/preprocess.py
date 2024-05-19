import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba  # Library for Chinese word segmentation
from collections import defaultdict
from multiprocessing import Pool

# nltk.download()

def data_preprocess(df,col):
    # 1. stop words
    # 2. TF-IDF
    chinese_stopwords = nltk.corpus.stopwords.words('chinese')
    lemma = WordNetLemmatizer()

    length = df.shape[0]
    data_without_stopwords = []

    # Loop through each reviews
    for i in range(0, length):
        reviews = df.iloc[i][col] # extract reviews
        #if reviews == None or str.strip(reviews) == '':
        #    continue
        doc = jieba.lcut(reviews.strip()) # Split the Chinese words

        doc = [lemma.lemmatize(word) for word in doc if not word in set(chinese_stopwords)]
        # remove characterists
        special_symbols = ['，','！','。','？','h','+', ' ', '、', '?','…','：',')','⊙','o','⊙','(',
                 '!',':','','...', "'"]
        doc = [value for value in doc if not value in set(special_symbols)]


        data_without_stopwords.append(doc)

    #
    word_count(data_without_stopwords)
    # I need to think about do I have to filter out the word that occurence less than 5, 10 or ...
    # to-do list:
    # 1. remove the word that occur less than ...
    df['cleaned reviews'] = data_without_stopwords

    data_without_stopwords = [item for t in data_without_stopwords for item in t]

    data_without_stopwords = remove_duplicates(data_without_stopwords)

    return df, data_without_stopwords

def word_count(word_lists):
    results = map_reduce(word_lists)
    # count the word frequency
    # export it to txt file
    with open('../word_frequency.txt', 'w') as f:
        # f.write(json.dumps(results))
        for key, value in results.items():
            f.write('%s:%s\n' % (key, value))

def mapper(words):
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word] +=1
    return word_freq

def reducer(word_freq_list):
    final_word_freq = defaultdict(int)
    for word_freq in word_freq_list:
        for word, freq in word_freq.items():
            final_word_freq[word] += freq
    return final_word_freq

def map_reduce(words_list):
    pool = Pool(processes= 4)
    mapped = pool.map(mapper, words_list)
    reduced = reducer(mapped)
    return reduced

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

def TF_IDF(words_list, n =100):
    # Import Tfidf vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(words_list)
    print("n_samples: %d, n_features: %d" % vectors.shape)

    # Select the first n documents from the data set
    tf_idf = pd.DataFrame(vectors.todense()).iloc[:n]
    tf_idf.columns = vectorizer.get_feature_names()
    tf_idf.to_csv('test.csv')
    tfidf_matrix = tf_idf.T
    tfidf_matrix.columns = ['response' + str(i) for i in range(1, n)]
    tfidf_matrix['count'] = tfidf_matrix.sum(axis=1)

    # Top 50 words
    tfidf_matrix = tfidf_matrix.sort_values(by='count', ascending=False)[:50]

    # Print the first 10 words
    print(tfidf_matrix.drop(columns=['count']).head(10))
