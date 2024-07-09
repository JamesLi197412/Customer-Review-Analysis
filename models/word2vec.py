import numpy as np
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def word2vec_gensim(words_list):
    model = Word2Vec(words_list, min_count=1)

    X = model[model.wv.vocab]
    words = list(model.wv.index_to_key)
    pca = PCA(n_components = 2)

    result = pca.fit_transform(X)

    # create a scatter plot of the project
    plt.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)

    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.show()



def generate_dictionary_data(text):
    word_to_index = dict()
    index_to_word = dict()
    corpus = []
    count = 0

    for row in text:
        for word in row:
            corpus.append(word)

            if word_to_index.get(word) == None:
                word_to_index |= {word:count}
                index_to_word |= {count:word}
                count+=1

    vocab_size = len(word_to_index)
    length_of_corpus = len(corpus)

    return word_to_index, index_to_word, corpus, vocab_size, length_of_corpus

def get_one_hot_vectors(target_word, context_words, vocab_size, word_to_index):
    trgt_word_vector = np.zeros(vocab_size)

    index_of_word_dictionary = word_to_index.get(target_word)

    # Set the index to 1
    trgt_word_vector[index_of_word_dictionary] = 1

    # Repeat same steps for context_words but in a loop
    ctxt_word_vector = np.zeros(vocab_size)

    for word in context_words:
        index_of_word_dictionary = word_to_index.get(word)
        ctxt_word_vector[index_of_word_dictionary] = 1

    return trgt_word_vector, ctxt_word_vector


def generate_training_data(corpus, window_size, vocab_size, word_to_index, length_of_corpus, sample=None):
    training_data = []
    training_sample_words = []

    for i, word in enumerate(corpus):
        index_target_word = i
        target_word = word
        context_words = []

        # when target word is the first word
        if i == 0:
            context_words = [corpus[x] for x in range(i + 1, window_size + 1)]

        elif i == len(corpus) - 1:
            context_words = [corpus[x] for x in range(length_of_corpus - 2, length_of_corpus - 2 - window_size, -1)]

        # When target word is the middle word
        else:
            # Before the middle target word
            before_target_word_index = index_target_word - 1
            for x in range(before_target_word_index, before_target_word_index - window_size, -1):
                if x >= 0:
                    context_words.extend([corpus[x]])

            # After the middle target word
            after_target_word_index = index_target_word + 1
            for x in range(after_target_word_index, after_target_word_index + window_size):
                if x < len(corpus):
                    context_words.extend([corpus[x]])

        trgt_word_vector, ctxt_word_vector = get_one_hot_vectors(target_word, context_words, vocab_size, word_to_index)
        training_data.append([trgt_word_vector, ctxt_word_vector])

        if sample is not None:
            training_sample_words.append([target_word, context_words])

    return training_data, training_sample_words



def forward_prop(weight_inp_hidden, weight_hidden_output, target_word_vector):
    # target_word_vector = x , weight_inp_hidden =  weights for input layer to hidden layer
    hidden_layer = np.dot(weight_inp_hidden.T, target_word_vector)

    # weight_hidden_output = weights for hidden layer to output layer
    u = np.dot(weight_hidden_output.T, hidden_layer)

    y_predicted = softmax(u)

    return y_predicted, hidden_layer, u


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def calculate_error(y_pred, context_words):
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}

    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update({index: 'yes'})

    number_of_1_in_context_vector = len(index_of_1_in_context_words)

    for i, value in enumerate(y_pred):

        if index_of_1_in_context_words.get(i) != None:
            total_error[i] = (value - 1) + ((number_of_1_in_context_vector - 1) * value)
        else:
            total_error[i] = (number_of_1_in_context_vector * value)

    return np.array(total_error)


def backward_prop(weight_inp_hidden, weight_hidden_output, total_error, hidden_layer, target_word_vector,
                  learning_rate):
    dl_weight_inp_hidden = np.outer(target_word_vector, np.dot(weight_hidden_output, total_error.T))
    dl_weight_hidden_output = np.outer(hidden_layer, total_error)

    # Update weights
    weight_inp_hidden = weight_inp_hidden - (learning_rate * dl_weight_inp_hidden)
    weight_hidden_output = weight_hidden_output - (learning_rate * dl_weight_hidden_output)

    return weight_inp_hidden, weight_hidden_output


def calculate_error(y_pred, context_words):
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}

    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update({index: 'yes'})

    number_of_1_in_context_vector = len(index_of_1_in_context_words)

    for i, value in enumerate(y_pred):

        if index_of_1_in_context_words.get(i) != None:
            total_error[i] = (value - 1) + ((number_of_1_in_context_vector - 1) * value)
        else:
            total_error[i] = (number_of_1_in_context_vector * value)

    return np.array(total_error)


def calculate_loss(u, ctx):
    sum_1 = 0
    for index in np.where(ctx == 1)[0]:
        sum_1 = sum_1 + u[index]

    sum_1 = -sum_1
    sum_2 = len(np.where(ctx == 1)[0]) * np.log(np.sum(np.exp(u)))

    total_loss = sum_1 + sum_2
    return total_loss


def train(word_embedding_dimension, window_size, epochs, training_data, learning_rate, disp='no', interval=-1):
    weights_input_hidden = np.random.uniform(-1, 1, (vocab_size, word_embedding_dimension))
    weights_hidden_output = np.random.uniform(-1, 1, (word_embedding_dimension, vocab_size))

    # For analysis purposes
    epoch_loss = []
    weights_1 = []
    weights_2 = []

    for epoch in range(epochs):
        loss = 0

        for target, context in training_data:
            y_pred, hidden_layer, u = forward_prop(weights_input_hidden, weights_hidden_output, target)

            total_error = calculate_error(y_pred, context)

            weights_input_hidden, weights_hidden_output = backward_prop(
                weights_input_hidden, weights_hidden_output, total_error, hidden_layer, target, learning_rate
            )

            loss_temp = calculate_loss(u, context)
            loss += loss_temp

        epoch_loss.append(loss)
        weights_1.append(weights_input_hidden)
        weights_2.append(weights_hidden_output)

        if disp == 'yes':
            if epoch == 0 or epoch % interval == 0 or epoch == epochs - 1:
                print('Epoch: %s. Loss:%s' % (epoch, loss))
    return epoch_loss, np.array(weights_1), np.array(weights_2)


# Input vector, returns nearest word(s)
def cosine_similarity(word, weight, word_to_index, vocab_size, index_to_word):
    # Get the index of the word from the dictionary
    index = word_to_index[word]

    # Get the correspondin weights for the word
    word_vector_1 = weight[index]

    word_similarity = {}

    for i in range(vocab_size):
        word_vector_2 = weight[i]

        theta_sum = np.dot(word_vector_1, word_vector_2)
        theta_den = np.linalg.norm(word_vector_1) * np.linalg.norm(word_vector_2)
        theta = theta_sum / theta_den

        word = index_to_word[i]
        word_similarity[word] = theta

    return word_similarity  # words_sorted


