import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# LDA evaluation
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.gensim_models as gensimvisualize
# reference: https://developer.ibm.com/tutorials/awb-lda-topic-modeling-text-analysis-python/#step-9-text-classification11

def LDA(words):
    # Load the dictionary
    dictionary = corpora.Dictionary(words)
    dictionary.filter_extremes(no_below = 2)

    # generate corpus as BoW
    corpus = [dictionary.doc2bow(word) for word in words]

    # train LDA model
    lda_model = LdaModel(corpus = corpus, id2word = dictionary, random_state = 4583, chunksize = 20, num_topics = 40,
                         passes = 200, iterations = 400)

    # print the LDA topics
    #for topic in lda_model.print_topics(num_topics = 40, num_words =10):
    #    print(topic)

    # Evaluate models
    coherence_model = CoherenceModel(model=lda_model, texts=words, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(coherence_score)

    # Visualize the topics
    dickens_visual = gensimvisualize.prepare(lda_model, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(dickens_visual,'/output/model_evaluation/lda.html')
    pyLDAvis.display(dickens_visual)

    # Text classification
    # generate document-topic distribution
    for i, doc in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc)
        print(f"Document {i}:{doc_topics}")
    return lda_model


