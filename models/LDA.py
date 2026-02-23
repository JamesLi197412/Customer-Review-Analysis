from pathlib import Path

import gensim.corpora as corpora
import pandas as pd
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvisualize
except Exception:
    pyLDAvis = None
    gensimvisualize = None


def _validate_token_lists(words):
    if not words:
        raise ValueError("Token list is empty. Run preprocessing before LDA.")
    if not any(words):
        raise ValueError("All tokenized reviews are empty after preprocessing.")


def build_dictionary_and_corpus(words, no_below=2, no_above=0.9):
    dictionary = corpora.Dictionary(words)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(word) for word in words]

    if len(dictionary) == 0:
        raise ValueError("Dictionary is empty after filtering. Adjust preprocessing or filter thresholds.")
    return dictionary, corpus


def train_lda_model(
    words,
    num_topics=20,
    no_below=2,
    no_above=0.9,
    passes=30,
    iterations=400,
    random_state=4583,
):
    _validate_token_lists(words)
    dictionary, corpus = build_dictionary_and_corpus(words, no_below=no_below, no_above=no_above)

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        random_state=random_state,
        chunksize=500,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations,
    )

    coherence_model = CoherenceModel(model=lda_model, texts=words, dictionary=dictionary, coherence="c_v")
    coherence_score = float(coherence_model.get_coherence())

    return lda_model, corpus, dictionary, coherence_score


def select_best_topic_count(
    words,
    topic_candidates=(8, 12, 16, 20),
    no_below=2,
    no_above=0.9,
    passes=30,
    iterations=400,
    random_state=4583,
):
    results = []
    best_result = None

    for topic_count in topic_candidates:
        model, corpus, dictionary, coherence = train_lda_model(
            words=words,
            num_topics=topic_count,
            no_below=no_below,
            no_above=no_above,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
        )
        row = {
            "num_topics": int(topic_count),
            "coherence_score": float(coherence),
        }
        results.append(row)

        if best_result is None or row["coherence_score"] > best_result["coherence_score"]:
            best_result = {
                "num_topics": int(topic_count),
                "coherence_score": float(coherence),
                "model": model,
                "corpus": corpus,
                "dictionary": dictionary,
            }

    evaluation_df = pd.DataFrame(results).sort_values("coherence_score", ascending=False).reset_index(drop=True)
    return best_result, evaluation_df


def save_lda_visualization(lda_model, corpus, dictionary, output_path="output/model_evaluation/lda.html"):
    if pyLDAvis is None or gensimvisualize is None:
        return False

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lda_visual = gensimvisualize.prepare(lda_model, corpus, dictionary, mds="mmds")
    pyLDAvis.save_html(lda_visual, str(output_file))
    return True


def topic_keywords_table(lda_model, num_words=10):
    rows = []
    for topic_idx in range(lda_model.num_topics):
        words = [word for word, _ in lda_model.show_topic(topic_idx, topn=num_words)]
        rows.append({"Dominant_Topic": topic_idx, "Topic_Keywords": ", ".join(words)})
    return pd.DataFrame(rows)


def LDA(words, num_topics=20):
    lda_model, corpus, dictionary, coherence_score = train_lda_model(words, num_topics=num_topics)
    save_lda_visualization(lda_model, corpus, dictionary, output_path="output/model_evaluation/lda.html")
    print(f"Coherence Score: {coherence_score:.4f}")
    return lda_model, corpus


def corpus_only(words):
    _, corpus = build_dictionary_and_corpus(words)
    return corpus


def format_topics_sentences(lda_model, corpus, data):
    dominant_rows = []
    for row in lda_model[corpus]:
        row = sorted(row, key=lambda x: x[1], reverse=True)
        if not row:
            dominant_rows.append(
                {
                    "Dominant_Topic": -1,
                    "Perc_Contribution": 0.0,
                    "Topic_Keywords": "",
                }
            )
            continue

        topic_num, prop_topic = row[0]
        topic_keywords = ", ".join([word for word, _ in lda_model.show_topic(topic_num)])
        dominant_rows.append(
            {
                "Dominant_Topic": int(topic_num),
                "Perc_Contribution": float(round(prop_topic, 4)),
                "Topic_Keywords": topic_keywords,
            }
        )

    sent_topics_df = pd.DataFrame(dominant_rows)
    return pd.concat([sent_topics_df, data.reset_index(drop=True)], axis=1)
