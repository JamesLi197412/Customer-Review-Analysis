import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def map_level_to_sentiment(level_id):
    try:
        level = int(level_id)
    except (TypeError, ValueError):
        return None

    if level <= 2:
        return "negative"
    if level == 3:
        return "neutral"
    return "positive"


def prepare_sentiment_dataset(df, text_col="normalized_text", label_col="LEVEL_ID"):
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' is missing.")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' is missing.")

    if "source_row" in df.columns:
        data = df[["source_row", text_col, label_col]].copy()
    else:
        data = df[[text_col, label_col]].copy()
        data["source_row"] = df.index
    data[text_col] = data[text_col].fillna("").astype(str)
    data["sentiment_label"] = data[label_col].apply(map_level_to_sentiment)
    data = data[data["sentiment_label"].notnull()]
    data = data[data[text_col].str.len() > 0]
    data["source_row"] = pd.to_numeric(data["source_row"], errors="coerce").fillna(-1).astype("int64")
    return data.reset_index(drop=True)


def _stratified_split(dataframe, label_col, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    test_indices = []

    for _, group in dataframe.groupby(label_col):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        if len(indices) <= 1:
            continue

        test_count = int(np.floor(len(indices) * test_size))
        test_count = max(1, test_count)
        test_count = min(test_count, len(indices) - 1)
        test_indices.extend(indices[:test_count].tolist())

    test_df = dataframe.loc[test_indices].copy()
    train_df = dataframe.drop(index=test_indices).copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed. Check label distribution and dataset size.")

    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def _build_classification_report(confusion_matrix, label_names):
    rows = []
    support_list = []
    f1_list = []

    for i, label_name in enumerate(label_names):
        tp = float(confusion_matrix[i, i])
        fp = float(confusion_matrix[:, i].sum() - tp)
        fn = float(confusion_matrix[i, :].sum() - tp)
        support = float(confusion_matrix[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        support_list.append(support)
        f1_list.append(f1_score)
        rows.append(
            {
                "label": label_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": int(support),
            }
        )

    total = float(confusion_matrix.sum())
    accuracy = float(np.trace(confusion_matrix) / total) if total > 0 else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    weighted_f1 = float(np.average(f1_list, weights=support_list)) if np.sum(support_list) > 0 else 0.0

    report_df = pd.DataFrame(rows)
    report_df = pd.concat(
        [
            report_df,
            pd.DataFrame(
                [
                    {"label": "accuracy", "precision": np.nan, "recall": np.nan, "f1_score": accuracy, "support": int(total)},
                    {"label": "macro avg", "precision": np.nan, "recall": np.nan, "f1_score": macro_f1, "support": int(total)},
                    {
                        "label": "weighted avg",
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1_score": weighted_f1,
                        "support": int(total),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    return report_df, accuracy, macro_f1, weighted_f1


def train_sentiment_classifier(
    df,
    text_col="normalized_text",
    label_col="LEVEL_ID",
    test_size=0.2,
    random_state=42,
    output_dir="output",
    max_tokens=30000,
    sequence_length=120,
    embedding_dim=128,
    epochs=8,
    batch_size=128,
):
    dataset = prepare_sentiment_dataset(df, text_col=text_col, label_col=label_col)
    if dataset.empty:
        raise ValueError("No valid rows available for sentiment training.")

    tf.keras.utils.set_random_seed(random_state)

    label_names = sorted(dataset["sentiment_label"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    dataset["sentiment_label_id"] = dataset["sentiment_label"].map(label_to_id).astype("int32")

    train_df, test_df = _stratified_split(
        dataset,
        label_col="sentiment_label_id",
        test_size=test_size,
        random_state=random_state,
    )

    x_train = train_df[text_col].astype(str).to_numpy()
    y_train = train_df["sentiment_label_id"].to_numpy()
    x_test = test_df[text_col].astype(str).to_numpy()
    y_test = test_df["sentiment_label_id"].to_numpy()

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorizer.adapt(x_train)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            vectorizer,
            tf.keras.layers.Embedding(max_tokens, embedding_dim, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(label_names), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
    predictions = probabilities.argmax(axis=1)

    confusion = tf.math.confusion_matrix(
        y_test,
        predictions,
        num_classes=len(label_names),
    ).numpy()
    report_df, accuracy, macro_f1, weighted_f1 = _build_classification_report(confusion, label_names)

    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{id_to_label[idx]}" for idx in range(len(label_names))],
        columns=[f"pred_{id_to_label[idx]}" for idx in range(len(label_names))],
    )

    prediction_samples_df = pd.DataFrame(
        {
            "true_label": [id_to_label[int(x)] for x in y_test],
            "pred_label": [id_to_label[int(x)] for x in predictions],
            "confidence": probabilities.max(axis=1).astype(float),
            "text_preview": pd.Series(x_test).str.slice(0, 120),
        }
    ).sort_values("confidence", ascending=False)

    all_probabilities = model.predict(dataset[text_col].astype(str).to_numpy(), batch_size=batch_size, verbose=0)
    all_prediction_ids = all_probabilities.argmax(axis=1)
    all_predictions_df = pd.DataFrame(
        {
            "source_row": dataset["source_row"].astype("int64"),
            "true_label": dataset["sentiment_label"],
            "pred_label": [id_to_label[int(x)] for x in all_prediction_ids],
            "confidence": all_probabilities.max(axis=1).astype(float),
        }
    )
    for label_id, label_name in id_to_label.items():
        all_predictions_df[f"prob_{label_name}"] = all_probabilities[:, int(label_id)]

    history_df = pd.DataFrame(history.history)

    base = Path(output_dir)
    analysis_dir = base / "analysis"
    models_dir = base / "models"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "sentiment_tf.keras"
    labels_path = models_dir / "sentiment_label_map.json"
    metrics_path = analysis_dir / "sentiment_metrics.json"
    report_path = analysis_dir / "sentiment_classification_report.csv"
    confusion_path = analysis_dir / "sentiment_confusion_matrix.csv"
    samples_path = analysis_dir / "sentiment_prediction_samples.csv"
    all_predictions_path = analysis_dir / "sentiment_all_predictions.csv"
    history_path = analysis_dir / "sentiment_training_history.csv"

    model.save(model_path)
    with labels_path.open("w", encoding="utf-8") as file:
        json.dump(id_to_label, file, ensure_ascii=False, indent=2)

    metrics = {
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "label_distribution": dataset["sentiment_label"].value_counts().to_dict(),
        "num_classes": int(len(label_names)),
        "epochs_trained": int(len(history_df)),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(confusion_path, encoding="utf-8-sig")
    prediction_samples_df.to_csv(samples_path, index=False, encoding="utf-8-sig")
    all_predictions_df.to_csv(all_predictions_path, index=False, encoding="utf-8-sig")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    return {
        "metrics": metrics,
        "model_path": str(model_path),
        "label_map_path": str(labels_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "confusion_path": str(confusion_path),
        "samples_path": str(samples_path),
        "all_predictions_path": str(all_predictions_path),
        "history_path": str(history_path),
    }
