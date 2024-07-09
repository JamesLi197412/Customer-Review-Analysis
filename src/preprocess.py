import re
from collections import defaultdict
from multiprocessing import Pool

import jieba  # Library for Chinese word segmentation
import nltk
from nltk.stem.wordnet import WordNetLemmatizer


def data_preprocess(df,col):
    chinese_stopwords = nltk.corpus.stopwords.words('chinese')
    lemma = WordNetLemmatizer()

    length = df.shape[0]
    data_without_stopwords = []

    # Loop through each review
    for i in range(0, length):
        reviews = df.iloc[i][col] # extract reviews
        doc = jieba.lcut(reviews.strip()) # Split the Chinese words

        doc = [lemma.lemmatize(word) for word in doc if not word in set(chinese_stopwords)]
        # remove special characterists
        special_symbols = ['，','！','。','？','h','+', ' ', '、', '?','…','：',')','⊙','o','⊙','(',
                 '!',':','','...', "'"]
        doc = [value for value in doc if not value in set(special_symbols)]


        data_without_stopwords.append(doc)


    df['cleaned reviews'] = data_without_stopwords
    #TF_IDF(data_without_stopwords, n=100)

    #data_without_stopwords = [item for t in data_without_stopwords for item in t]
    # print(data_without_stopwords)
    #data_without_stopwords = remove_duplicates(data_without_stopwords)

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


