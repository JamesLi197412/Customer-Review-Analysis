import multiprocessing

from gensim.models import Phrases
from gensim.models import Word2Vec

# Word2Vec -- Skil-grams & Continuous-bag-of-words.
def word2vec_gensim(words_list):
    bigram_transformer = Phrases(words_list)

    cores = multiprocessing.cpu_count()
    model = Word2Vec(bigram_transformer[words_list],
                     min_count=5,
                     vector_size=100,
                     window=5,
                     workers=cores - 1,
                     epochs=200)
    # save the model
    model.save("word2vec.model")

    return model


def word2vec_run(data, word, target):
    model = word2vec_gensim(data)
    # Test
    # print(model.mv.most_similar(positive = [word]))

    # Similarity test
    result = model.wv.similarity(word, target)

    return result
