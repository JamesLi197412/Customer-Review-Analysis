import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from exploration.description import EDA, data_process, visulaisation
from src.preprocess import data_preprocess, word_count


def parse_args():
    parser = argparse.ArgumentParser(description="Customer feedback NLP pipeline")
    parser.add_argument("--input-path", default="data/CUSTOMER_FEEDBACK.xlsx")
    parser.add_argument("--sheet-name", default="Sheet")
    parser.add_argument("--text-col", default="CONTENT_TX")
    parser.add_argument("--time-col", default="SURVEY_TIME")
    parser.add_argument("--label-col", default="LEVEL_ID")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--stopwords-path", default="data/chinese_stopwords.txt")
    parser.add_argument("--topic-candidates", default="8,12,16,20")
    parser.add_argument("--top-topic-reviews", type=int, default=5)
    parser.add_argument("--run-visualization", action="store_true")
    parser.add_argument("--skip-lda", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--hybrid-priority-topn", type=int, default=10)
    return parser.parse_args()


def ensure_output_dirs(base_output):
    base = Path(base_output)
    paths = {
        "base": base,
        "analysis": base / "analysis",
        "model_eval": base / "model_evaluation",
        "models": base / "models",
        "visualisation": base / "visualisation",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def parse_topic_candidates(raw_value):
    values = []
    for value in str(raw_value).split(","):
        value = value.strip()
        if not value:
            continue
        try:
            values.append(int(value))
        except ValueError:
            continue
    return tuple(sorted(set(values))) if values else (8, 12, 16, 20)


def remove_duplicates(words_list):
    return list(dict.fromkeys(words_list))


def top_sentence_topic(df_topic_sents_keywords, n=5):
    topics_reviews = {}
    topics = df_topic_sents_keywords["Topic_Keywords"].dropna().unique()
    for topic in topics:
        topic_sub = df_topic_sents_keywords[df_topic_sents_keywords["Topic_Keywords"] == topic].copy(deep=True)
        topic_sub = topic_sub.sort_values("Perc_Contribution", ascending=False).head(n)
        content_lists = list(topic_sub["cleaned reviews"])
        content = remove_duplicates([item for tokens in content_lists for item in tokens])
        topics_reviews[topic] = content

    return pd.DataFrame(topics_reviews.items(), columns=["Topics", "Comments"])


def _safe_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def build_hybrid_outputs(
    lda_assignments_path,
    sentiment_predictions_path,
    output_paths,
    top_n=10,
):
    lda_df = pd.read_csv(lda_assignments_path)
    sentiment_df = pd.read_csv(sentiment_predictions_path)

    if "source_row" not in lda_df.columns:
        lda_df = lda_df.reset_index(drop=True).copy()
        lda_df["source_row"] = lda_df.index
    if "source_row" not in sentiment_df.columns:
        sentiment_df = sentiment_df.reset_index(drop=True).copy()
        sentiment_df["source_row"] = sentiment_df.index

    merged_df = lda_df.merge(
        sentiment_df[["source_row", "pred_label", "confidence", "true_label"]],
        how="left",
        on="source_row",
    )

    merged_path = output_paths["analysis"] / "hybrid_review_level.csv"
    merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")

    grouped_rows = []
    for (topic_id, topic_keywords), group in merged_df.groupby(["Dominant_Topic", "Topic_Keywords"], dropna=False):
        volume = int(len(group))
        if volume == 0:
            continue

        pred_labels = group["pred_label"].fillna("unknown")
        negative_count = int((pred_labels == "negative").sum())
        neutral_count = int((pred_labels == "neutral").sum())
        positive_count = int((pred_labels == "positive").sum())
        unknown_count = int((pred_labels == "unknown").sum())

        if "LEVEL_ID" in group.columns:
            level_numeric = _safe_numeric_series(group["LEVEL_ID"])
            low_rating_ratio = float((level_numeric <= 2).mean())
            avg_level = float(level_numeric.mean())
        else:
            low_rating_ratio = 0.0
            avg_level = 0.0

        negative_ratio = negative_count / volume
        confidence_mean = float(_safe_numeric_series(group["confidence"]).mean())
        avg_topic_weight = float(_safe_numeric_series(group["Perc_Contribution"]).mean())

        # Priority score emphasizes negative sentiment and low-rating proportion.
        priority_score = (0.7 * negative_ratio + 0.3 * low_rating_ratio) * (1.0 + min(volume / 1000.0, 1.0))

        grouped_rows.append(
            {
                "Dominant_Topic": int(topic_id),
                "Topic_Keywords": topic_keywords,
                "volume": volume,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_count": positive_count,
                "unknown_count": unknown_count,
                "negative_ratio": float(negative_ratio),
                "low_rating_ratio": float(low_rating_ratio),
                "avg_level_id": float(avg_level),
                "avg_confidence": confidence_mean,
                "avg_topic_contribution": avg_topic_weight,
                "priority_score": float(priority_score),
            }
        )

    if not grouped_rows:
        hybrid_summary_df = pd.DataFrame(
            columns=[
                "Dominant_Topic",
                "Topic_Keywords",
                "volume",
                "negative_count",
                "neutral_count",
                "positive_count",
                "unknown_count",
                "negative_ratio",
                "low_rating_ratio",
                "avg_level_id",
                "avg_confidence",
                "avg_topic_contribution",
                "priority_score",
            ]
        )
    else:
        hybrid_summary_df = pd.DataFrame(grouped_rows).sort_values(
            by=["priority_score", "negative_ratio", "volume"],
            ascending=[False, False, False],
        )

    summary_path = output_paths["analysis"] / "hybrid_topic_sentiment_summary.csv"
    priority_path = output_paths["analysis"] / "hybrid_priority_topics.csv"
    hybrid_summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    hybrid_summary_df.head(max(1, top_n)).to_csv(priority_path, index=False, encoding="utf-8-sig")

    return {
        "review_level_path": str(merged_path),
        "topic_summary_path": str(summary_path),
        "priority_topics_path": str(priority_path),
        "topic_count": int(len(hybrid_summary_df)),
    }


def run_lda_pipeline(df, tokens, args, output_paths):
    from models.LDA import format_topics_sentences, save_lda_visualization, select_best_topic_count

    topic_candidates = parse_topic_candidates(args.topic_candidates)
    best_result, topic_eval_df = select_best_topic_count(tokens, topic_candidates=topic_candidates)

    lda_model = best_result["model"]
    corpus = best_result["corpus"]
    dictionary = best_result["dictionary"]
    best_coherence = best_result["coherence_score"]
    best_topic_count = best_result["num_topics"]

    topic_eval_path = output_paths["analysis"] / "lda_topic_search.csv"
    topic_eval_df.to_csv(topic_eval_path, index=False, encoding="utf-8-sig")

    visualization_path = output_paths["model_eval"] / "lda.html"
    save_lda_visualization(lda_model, corpus, dictionary, output_path=str(visualization_path))

    model_path = output_paths["models"] / "lda_model.joblib"
    joblib.dump(lda_model, model_path)

    topic_df = format_topics_sentences(lda_model, corpus, df)
    topic_columns = ["source_row", "Dominant_Topic", "Perc_Contribution", "Topic_Keywords", args.text_col, args.label_col, "cleaned reviews"]
    topic_df = topic_df.reset_index(drop=True)[topic_columns].copy(deep=True)

    topic_reviews = top_sentence_topic(topic_df, n=args.top_topic_reviews)

    topic_assignments_path = output_paths["analysis"] / "lda_topic_assignments.csv"
    topic_summary_path = output_paths["analysis"] / "lda_topic_summary.csv"
    topic_df.to_csv(topic_assignments_path, index=False, encoding="utf-8-sig")
    topic_reviews.to_csv(topic_summary_path, index=False, encoding="utf-8-sig")

    return {
        "best_topic_count": int(best_topic_count),
        "best_coherence": float(best_coherence),
        "topic_search_path": str(topic_eval_path),
        "topic_assignments_path": str(topic_assignments_path),
        "topic_summary_path": str(topic_summary_path),
        "model_path": str(model_path),
        "visualization_path": str(visualization_path),
    }


def run_pipeline(args):
    output_paths = ensure_output_dirs(args.output_dir)

    customer_feedback = pd.read_excel(args.input_path, sheet_name=args.sheet_name)
    customer_feedback = customer_feedback.reset_index(drop=True)
    customer_feedback["source_row"] = customer_feedback.index
    customer_feedback = data_process(customer_feedback, args.time_col, args.text_col)
    customer_feedback = EDA(customer_feedback, output_path=output_paths["analysis"] / "data_profile.json")

    if args.run_visualization:
        visulaisation(customer_feedback, output_dir=output_paths["visualisation"], show=False)

    customer_feedback, words_list = data_preprocess(
        customer_feedback,
        args.text_col,
        stopwords_path=args.stopwords_path,
    )
    word_count(words_list, output_path=output_paths["analysis"] / "word_frequency.txt")

    preprocessed_path = output_paths["analysis"] / "preprocessed_feedback.csv"
    customer_feedback.to_csv(preprocessed_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_path": args.input_path,
        "rows": int(customer_feedback.shape[0]),
        "output_dir": str(output_paths["base"]),
        "preprocessed_path": str(preprocessed_path),
    }

    if not args.skip_lda:
        try:
            summary["lda"] = run_lda_pipeline(customer_feedback, words_list, args, output_paths)
        except ModuleNotFoundError as error:
            missing_pkg = getattr(error, "name", "unknown")
            raise ModuleNotFoundError(
                f"LDA pipeline dependency missing: '{missing_pkg}'. "
                "Install dependencies with `pip install -r requirements.txt` or run with `--skip-lda`."
            ) from error

    if not args.skip_sentiment:
        try:
            from models.sentiment import train_sentiment_classifier

            summary["sentiment"] = train_sentiment_classifier(
                customer_feedback,
                text_col="normalized_text",
                label_col=args.label_col,
                output_dir=args.output_dir,
            )
        except ModuleNotFoundError as error:
            missing_pkg = getattr(error, "name", "unknown")
            raise ModuleNotFoundError(
                f"Sentiment pipeline dependency missing: '{missing_pkg}'. "
                "Install dependencies with `pip install -r requirements.txt` or run with `--skip-sentiment`."
            ) from error

    if "lda" in summary and "sentiment" in summary:
        summary["hybrid"] = build_hybrid_outputs(
            lda_assignments_path=summary["lda"]["topic_assignments_path"],
            sentiment_predictions_path=summary["sentiment"]["all_predictions_path"],
            output_paths=output_paths,
            top_n=args.hybrid_priority_topn,
        )

    summary_path = output_paths["analysis"] / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Pipeline complete. Summary saved to: {summary_path}")
    return summary


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(cli_args)
