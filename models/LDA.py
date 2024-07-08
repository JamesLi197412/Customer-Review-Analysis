import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd

# LDA evaluation
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.gensim_models as gensimvisualize
# reference: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# reference: https://developer.ibm.com/tutorials/awb-lda-topic-modeling-text-analysis-python/#step-9-text-classification11

def LDA(words, num_topics = 10):
    # Load the dictionary
    dictionary = corpora.Dictionary(words)
    dictionary.filter_extremes(no_below = 2)

    # generate corpus as BoW
    corpus = [dictionary.doc2bow(word) for word in words]
    # Export corpus locally in the future.
    write_list(corpus)

    # train LDA model
    lda_model = LdaModel(corpus = corpus, id2word = dictionary, random_state = 4583, chunksize = 20, num_topics = num_topics,
                         passes = 200, iterations = 100)

    # Evaluate models
    coherence_model = CoherenceModel(model=lda_model, texts=words, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(coherence_score)

    # Visualize the topics
    dickens_visual = gensimvisualize.prepare(lda_model, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(dickens_visual,'output/model_evaluation/lda.html')
    pyLDAvis.display(dickens_visual)

    return lda_model,corpus


def format_topics_sentences(lda_model, corpus, words):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(words)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def write_list(lists):
    with open(r'corpus.txt', 'w') as fp:
        for item in lists:
            # write each item on a new line
            fp.write("%s\n" % item)



