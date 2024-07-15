from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# import keras
#from keras.models import Sequential
#from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Embedding, LSTM, Dense
#from sklearn.metrics import accuracy_score


# https://www.analyticsvidhya.com/blog/2022/01/sentiment-analysis-with-lstm/
def lstm_model(df):
    # vocab_size - 3000, embedding_dim = 100, max_length = 200
    reviews = df['Topic_Keywords'].values
    labels = df['LEVEL_ID'].values

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)


    train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels,
                                                                                  stratify=encoded_labels)

    max_words = 1000
    embedding_dim = 100
    max_len = 100
    # -- Build the Model
    # model initialization
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length = max_len))
    model.add(LSTM(64))
    model.add(Dense(2, activation= 'softmax'))
    """
    model = keras.Sequential([
        keras.layers.Embedding(input_dim = len(train_sentences), output_dim = 200),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    """
    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model summary
    print(model.summary())
    return model, train_sentences, test_sentences, train_labels, test_labels

def train_model(model, train_sentences, train_labels):
    num_epochs = 5
    history = model.fit(train_sentences, train_labels,
                        epochs=num_epochs, verbose=1,
                        validation_split=0.1)

def evaulation(model, test_df,test_labels):
    prediction = model.predict(test_df)

    # Evaulation
    # Get labels based on probability 1 if p>= 0.5 else 0
    pred_labels = []
    for i in prediction:
        if i >= 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    print("Accuracy of prediction on test set : ", accuracy_score(test_labels, pred_labels))