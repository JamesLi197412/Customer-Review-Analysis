import tensorflow as tf


def build_lstm_text_classifier(num_classes=3, max_tokens=30000, sequence_length=120, embedding_dim=128):
    text_vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            text_vectorizer,
            tf.keras.layers.Embedding(max_tokens, embedding_dim, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, text_vectorizer
