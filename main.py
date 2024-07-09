import warnings

import joblib

from exploration.description import *
from models.LDA import *
from models.tf_idf import *
from models.word2vec import *
from src.preprocess import *


def warn(*args, **kwargs):
    pass

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
    lda_model, corpus = LDA(words_list, num_topics=10)

    joblib.dump(lda_model, filename)
    return lda_model, corpus

def load_model():
    loaded_model = joblib.load('lda_model.sav')

    return loaded_model

def top_sentence_topic(df_topic_sents_keywords):
    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    print(sent_topics_outdf_grpd.head(5))

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Perc_Contribution", "Topic_Keywords", "CONTENT_TX"]

    return sent_topics_sorteddf_mallet

def word2vec_run(data, window_size):
    word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictionary_data(data)
    print(corpus)
    #training_data, training_sample_words = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'Yes')
    #np.savetxt('myarray.txt', training_data)






if __name__ == '__main__':
    # input your data with relative path
    warnings.warn = warn()

    customer_feedback = pd.read_excel('data/CUSTOMER_FEEDBACK.xlsx', sheet_name='Sheet')

    # Column Adjustment
    customer_feedback = data_process(customer_feedback,'SURVEY_TIME','CONTENT_TX')
    # visulaisation(customer_feedback)  # generate visualisation and export it output folder

    customer_feedback = EDA(customer_feedback)

    # word process -- > add cleaned reviews columns
    customer_feedback, words_list = data_preprocess(customer_feedback, 'CONTENT_TX')  # words_list -- list of list
    customer_feedback.to_csv('test.csv')
    '''
    # Export file to view its output
    file_export(customer_feedback, words_list)

    # Call LDA model method
    lda_model, corpus = lda_operation(words_list, 'lda_model.sav')
    # lda_model = load_model()
    # corpus = corpus_only(words_list)
    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, customer_feedback)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic_sub = df_topic_sents_keywords[
        ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'CONTENT_TX']].copy(deep=True)

    sent_topics_sorteddf_mallet = top_sentence_topic(df_dominant_topic_sub)
    '''
    # TF_IDF(words_list, n=5)

    # Run word2vec
    # word2vec_run(data = words_list, window_size=2)
    word2vec_gensim(words_list)

