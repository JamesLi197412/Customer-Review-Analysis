### Project Name
Chinese Customer Feedback Intelligence (NLP)

### One-line Summary
Built a reproducible NLP pipeline to convert 30K+ Chinese customer reviews into actionable topics and measurable sentiment insights.

### Business Goal
Identify what drives low customer ratings and provide store-operation improvement directions by extracting themes and sentiment from unstructured feedback text.

### What I Built
1. Data engineering and EDA
- Parsed timestamp into multiple features (year/month/day/hour)
- Generated missing-value and schema profile outputs for auditability

2. Chinese NLP preprocessing
- Implemented robust tokenization via `jieba`
- Added stopword/symbol filtering and null-safe preprocessing

3. Topic intelligence
- Trained LDA topic models across multiple topic counts
- Selected the best model using coherence score
- Exported dominant topic labels per review and topic keyword summaries

4. Sentiment modeling
- Converted 1-5 score labels into negative/neutral/positive classes
- Trained a TensorFlow text classifier (TextVectorization + BiLSTM)
- Exported metrics, classification report, confusion matrix, prediction samples, full-review predictions, and training history

5. Hybrid decision layer (final solution)
- Merged LDA topic assignment with sentiment predictions at review level
- Ranked topics by hybrid priority score (negative ratio + low-rating ratio + volume weight)
- Produced business-facing priority list of issues for action planning

6. Reproducibility
- Refactored into a configurable CLI pipeline
- Standardized artifact outputs under versioned folders

### Metrics To Fill In
- Best LDA coherence score: `<fill from output/analysis/lda_topic_search.csv>`
- Sentiment accuracy: `<fill from output/analysis/sentiment_metrics.json>`
- Weighted F1: `<fill from output/analysis/sentiment_metrics.json>`
- Number of reviews processed: `<fill from output/analysis/pipeline_summary.json>`
