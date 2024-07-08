from exploration.description import *
from src.preprocess import *
from models.LDA import *

import warnings
import joblib


# Reference:
# 1. https://zhuanlan.zhihu.com/p/86579316
# 2. https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/


def warn(*args, **kwargs):
    pass


def file_export(df, text):
    df.to_csv('test.csv')

    with open(r'words.txt','w') as fp:
        fp.write("\n".join(str(word) for word in text))

def model_store(model,filename):
    # To save model locally
    filename = 'lda_model.sav'
    joblib.dump(model, lda_filename)

def load_model(model,filename):
    pass

def top_sentence_topic():
    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    # Show
    sent_topics_sorteddf_mallet.head()
    return sent_topics_sorteddf_mallet

if __name__ == '__main__':
    # input your data with relative path
    warnings.warn = warn()
    customer_feedback = pd.read_excel('data/CUSTOMER_FEEDBACK.xlsx', sheet_name='Sheet')

    # Column Adjustment
    customer_feedback = data_process(customer_feedback,'SURVEY_TIME','CONTENT_TX')
    # visulaisation(customer_feedback)  # generate visualisatoin output to folder

    customer_feedback = EDA(customer_feedback)

    # word process -- > add cleaned reviews columns
    customer_feedback,words_list = data_preprocess(customer_feedback, 'CONTENT_TX')

    # Export file to view its output
    file_export(customer_feedback, words_list)

    # Call LDA model method
    lda_model,corpus = LDA(words_list, num_topics=15)

    # To save model locally
    lda_filename = 'lda_model.sav'
    joblib.dump(lda_model, lda_filename)

    df_topic_sents_keywords = format_topics_sentences(lda_model,corpus, words_list)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    df_dominant_topic.head(10)










