from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# import keras
# from keras.models import Sequential
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Embedding, LSTM, Dense
# from sklearn.metrics import accuracy_score

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

#
#     model = keras.Sequential([
#         keras.layers.Embedding(input_dim = len(train_sentences), output_dim = 200),
#         keras.layers.Bidirectional(keras.layers.LSTM(64)),
#         keras.layers.Dense(2, activation='relu'),
#         keras.layers.Dense(1, activation='sigmoid')
#     ])
#
#     return model, train_sentences, test_sentences, train_labels, test_labels
#
# def train_model(model, train_sentences, train_labels):
#     num_epochs = 5
#     history = model.fit(train_sentences, train_labels,
#                         epochs=num_epochs, verbose=1,
#                         validation_split=0.1)
#
# def evaulation(model, test_df,test_labels):
#     prediction = model.predict(test_df)
#
#     # Evaulation
#     # Get labels based on probability 1 if p>= 0.5 else 0
#     pred_labels = []
#     for i in prediction:
#         if i >= 0.5:
#             pred_labels.append(1)
#         else:
#             pred_labels.append(0)
