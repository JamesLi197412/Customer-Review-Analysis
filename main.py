import warnings
import joblib
import pandas as pd

from exploration.description import *
from models.LDA import *
from models.tf_idf import *
from models.word2vec import *
#from models.LSTM import *
from src.preprocess import *


def read_lists():
    corpus_list = []
    with open(r'corpus.txt', 'r') as f:
        for line in f:
            x = (line[:-1])
            corpus_list.append(x)

    return corpus_list

def file_export(df, text):
    df.to_csv('test.csv')

    with open(r'words.txt','w') as fp:
        fp.write("\n".join(str(word) for word in text))

def lda_operation(words_list,filename):
    # Call LDA model method
    lda_model, corpus = LDA(words_list, num_topics=20)

    joblib.dump(lda_model, filename)
    return lda_model, corpus

def load_model():
    loaded_model = joblib.load('lda_model.sav')

    return loaded_model

def remove_duplicates(words_list):
    temp = []
    for x in words_list:
        if x not in temp:
            temp.append(x)

    return temp

def top_sentence_topic(df_topic_sents_keywords,N = 5):
    topics_reviews = {}

    # Loop through each main topic
    topics = df_topic_sents_keywords['Topic_Keywords'].unique()
    for topic in topics:
        topic_sub = df_topic_sents_keywords[df_topic_sents_keywords['Topic_Keywords'] == topic].copy(deep = True)
        topic_sub = topic_sub.sort_values('Perc_Contribution').head(N)
        content_lists = list(topic_sub['cleaned reviews'])
        content = remove_duplicates([item for t in content_lists for item in t])

        topics_reviews[topic] = content

    topics_reviews_df = pd.DataFrame(topics_reviews.items(), columns = ['Topics', 'Comments'])
    return topics_reviews_df

def topic_modelling_lda(words_list, option = 2):
    if option == 1:
        # when you call the function to train the model
        lda_model, corpus = lda_operation(words_list, 'lda_model.sav')
    else:
        lda_model = load_model()
        corpus = corpus_only(words_list)

    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, customer_feedback)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic_sub = df_dominant_topic[
        ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'CONTENT_TX','LEVEL_ID','cleaned reviews']].copy(deep=True)

    # Evaulation
    topics_review = top_sentence_topic(df_dominant_topic_sub, N = 5)
    return df_dominant_topic_sub, topics_review


if __name__ == '__main__':
    # input your data with relative path
    customer_feedback = pd.read_excel('data/CUSTOMER_FEEDBACK.xlsx', sheet_name='Sheet')

    # Column Adjustment
    customer_feedback = data_process(customer_feedback,'SURVEY_TIME','CONTENT_TX')
    # visulaisation(customer_feedback)  # generate visualisation and export it output folder

    customer_feedback = EDA(customer_feedback)

    # word process -- > add cleaned reviews columns
    customer_feedback, words_list = data_preprocess(customer_feedback, 'CONTENT_TX')  # words_list -- list of list


    # Export file to view its output
    file_export(customer_feedback, words_list)

    # LDA modelling
    df_dominant_topic_sub,topic_reviews = topic_modelling_lda(words_list, 2)
    topic_reviews.to_csv('result.csv',encoding='utf-8-sig')
    # lstm_model(df_dominant_topic_sub, vocab_size = 100)





