# Predicting GME Stock Price Movement Using Multi-Model Sentiment Analysis of r/wallstreetbets

**Michael Koach**
Professional Certificate in Machine Learning and Artificial Intelligence, UC Berkeley

---

## Executive Summary

This project compares five sentiment analysis approaches for predicting daily GME stock price direction using Reddit posts from r/wallstreetbets during the January to March 2021 short squeeze:

- **VADER** (rule-based, general purpose)
- **FinBERT** (transformer, financial domain)
- **Twitter-RoBERTa** (transformer, social media domain)
- **Topic-Sentiment RoBERTa** (transformer, entity-targeted)
- **GoEmotions** (transformer, 28 emotion labels trained on Reddit)

The study extends Wang & Luo (2021), who used VADER as their sole sentiment tool and reported strong classification results when combined with semantic embeddings. We replicate their preprocessing pipeline and test whether modern transformer-based models, each adapted to a different domain, can improve sentiment-based prediction of stock price direction.

After aggregating post-level features to the daily level (n=44 trading days) and applying time-series cross-validation, we find that no sentiment model or feature engineering technique reliably predicts daily price direction beyond a majority-class baseline of 59.1%. As a secondary finding, we identify a data leakage issue in the original paper's train/test split methodology that likely inflated their reported accuracy. Under corrected methodology, all models produce near-chance accuracy at the post level, and modest improvements over baseline at the daily level.

---

## Research Question

Do modern transformer-based sentiment models outperform the rule-based VADER approach for predicting the daily direction of GME's stock price using sentiment extracted from r/wallstreetbets?

---

## Background and Motivation

In late January 2021, GameStop's stock price experienced unprecedented volatility as retail investors coordinated through the r/wallstreetbets subreddit to execute a short squeeze against institutional short sellers. The stock rose from under $20 to a pre-market peak of over $500 per share, drawing global media attention and raising questions about social media's influence on financial markets.

Wang & Luo (2021) investigated this relationship by using VADER sentiment analysis combined with word2vec and BERT semantic embeddings to predict the binary direction (up/down) of GME's daily net price movement from r/wallstreetbets posts. Their best configuration achieved 99.05% accuracy using a Decision Tree classifier with word2vec embeddings and VADER sentiment features. However, they also found that VADER sentiment alone performed poorly, with two of four classifiers producing F1 scores of zero for the minority class. Their own discussion acknowledged an inability to "definitively demonstrate a strong relationship between sentiment and price movement," and a preliminary annotation experiment revealed that VADER agreed with human annotators at only κ = 0.078 on r/wallstreetbets text.

These findings raise a natural question: would more sophisticated sentiment tools overcome the limitations that VADER faced on this uniquely challenging corpus? The paper identified three primary obstacles to sentiment analysis on r/wallstreetbets text: frequent sarcasm and irony (where profanity conveys positive sentiment), out-of-vocabulary slang and investment jargon, and non-standard word usages specific to the subreddit. Our multi-model approach directly tests whether domain-adapted transformers can handle these challenges where VADER could not.

This project extends the original work in three ways. First, we replace the single VADER baseline with five sentiment models spanning different domains and architectures. Second, we aggregate features to the daily level and use time-series cross-validation to prevent posts from the same trading day from appearing in both training and test sets. Third, we test whether engagement-weighted sentiment (weighting by upvotes) outperforms equal-weight sentiment.

---

## Data Sources

| Source | Description | Details |
|--------|-------------|---------|
| [Reddit WallStreetBets Posts](https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts) | WSB post titles, bodies, scores, and comment counts | 35,735 posts, Jan 28 to Mar 31, 2021 |
| [Yahoo Finance (yfinance)](https://pypi.org/project/yfinance/) | GME daily open/close prices | 44 trading days, Jan 28 to Mar 31, 2021 |

**Note on dataset discrepancy:** Wang & Luo (2021) report using the same Kaggle dataset, filtering for January 4 to March 31, 2021 (35,726 posts). Our download of the identical dataset yields 35,735 posts beginning January 28, 2021. The Kaggle dataset appears to have been modified since the paper's publication, with the earliest available posts now starting January 28 rather than January 4. The missing early period (Jan 4 to 27) represents the buildup phase before the squeeze peaked. Despite this, the total post counts are nearly identical (35,735 vs. 35,726), the core squeeze and aftermath are fully captured, and the direction distribution closely matches the paper's reported figures (20.5% up vs. 21.2% up). This discrepancy does not materially affect our analysis but is documented for reproducibility.

---

## Methodology

### Text Preprocessing

The preprocessing pipeline replicates the methods outlined in the reference paper: title and body concatenation, emoji replacement with descriptive text tags using pipe delimiters (e.g., `|rocket|`), URL removal, Reddit user mention removal, punctuation removal (preserving decimal numbers and emoji tags), and whitespace normalization. We also apply lowercasing as a standard preprocessing step. The reference paper does not explicitly state whether they lowercased text before sentiment analysis. Because VADER is case-sensitive (all-caps text boosts sentiment intensity scores), and all-caps posts are common on r/wallstreetbets, our lowercasing step may slightly reduce VADER's sensitivity to emphasis. This is noted for transparency but applies equally across all experiments.

### Sentiment Models

Five sentiment models are used, each testing a distinct hypothesis about domain adaptation for financial social media text:

| Model | Type | Domain | Outputs | Rationale |
|-------|------|--------|---------|-----------|
| **VADER** | Rule-based | General social media | pos, neu, neg, compound | Paper's original baseline (paper used pos, neu, neg only; we add compound); known poor performance on WSB text (κ = 0.078) |
| **FinBERT** | Transformer | Financial text | positive, neutral, negative | Tests whether financial domain adaptation improves prediction |
| **Twitter-RoBERTa** | Transformer | Social media (124M tweets) | positive, neutral, negative | Tests whether social media domain adaptation captures WSB language |
| **Topic-Sentiment RoBERTa** | Transformer | Social media (entity-targeted) | 5-point scale | Tests whether GME-targeted sentiment outperforms general sentiment |
| **GoEmotions** | Transformer | Reddit (28 emotion labels) | 28 emotions + neutral | Tests whether granular emotion classification captures dynamics that polarity models miss |

### Data Leakage Identification

The reference paper trained classifiers at the post level (n = 35,726) using a stratified 80/20 train/test split. Because all posts on a given trading day share the same Direction label, this split allows posts from the same day to appear in both training and test sets. The classifier can learn day-specific textual patterns (vocabulary, writing style, topics discussed on a particular day) and then recognize those same patterns in test posts from the same day. This inflates apparent accuracy.

The effective sample size for this prediction task is not 35,726 posts but 44 trading days. Posts are not independent observations; they are clustered by day with a shared label.

### Daily Aggregation

To address this, we aggregate all post-level features to the daily level, producing one row per trading day (n = 44). Two aggregation strategies are tested:

- **Unweighted:** Simple mean and standard deviation of all sentiment and text features per day
- **Engagement-weighted:** Sentiment features weighted by upvote score, under the hypothesis that highly upvoted posts better represent community sentiment. Metadata and text features use unweighted aggregation to avoid circular weighting (weighting upvotes by upvotes).

### Feature Engineering

Several feature engineering techniques address the high feature-to-observation ratio (up to 50 features on 44 rows):

- **Correlation-based selection:** Top 5 and top 10 features ranked by absolute Pearson correlation with daily net price movement
- **PCA dimensionality reduction:** GoEmotions compressed from 28 to 12 components, all sentiment features from 41 to 15 components, and top-10 correlated features from 10 to 6 components. Component counts were determined by scree plot analysis at the 90% cumulative variance threshold.
- **Composite features:** Hand-engineered features including polarization scores (1 minus neutral), bullish ratios (positive divided by positive plus negative), GoEmotions-derived solidarity and cognitive indices, and log-transformed post count and engagement metrics
- **Sentiment dispersion:** Standard deviations of the top-correlated features, capturing within-day sentiment agreement

### Classification Approach

Six classifiers are trained: Logistic Regression, SVM, Decision Tree, Random Forest, MLP, and XGBoost. These include the four classifiers used in the reference paper (SVM, RF, DT, MLP) plus Logistic Regression and XGBoost.

All classifiers are wrapped in a StandardScaler pipeline and tuned via GridSearchCV with 3-fold TimeSeriesSplit cross-validation. Regularization is applied aggressively given the small sample size: strong L1/L2 penalties on linear models, depth constraints on tree-based models, small hidden layers on MLP, and L1/L2 regularization on XGBoost.

The majority-class baseline (always predicting "down") achieves 59.1% accuracy. Accuracy is used as the primary metric to match the reference paper, with fold-level standard deviations reported to assess stability.

### Note on Semantic Embeddings

The reference paper's best result (99.05% accuracy) used word2vec semantic embeddings alongside VADER sentiment. We do not replicate the embedding experiments for two reasons. First, our research question focuses specifically on comparing sentiment analysis techniques, and embeddings are general semantic representations rather than sentiment measures. Second, word2vec (100 dimensions) and BERT (768 dimensions) embeddings on 44 daily observations would create severe overfitting. The paper's use of embeddings was only feasible at the post level (n = 35,726), which, as discussed, suffers from data leakage.

---

## Exploratory Data Analysis

Detailed EDA is presented in Notebook 2 (EDA.ipynb). Key findings:

**Post volume trails price movement.** Reddit activity spikes after large price swings rather than before them, suggesting the community was largely reactive rather than predictive. January 29 alone accounts for approximately 44% of all posts, creating extreme skew in the post-level dataset.

**Class balance changes dramatically with aggregation.** At the post level, classes appear severely imbalanced (79.5% down, 20.5% up). At the daily level, the split is much more balanced (59.1% down, 40.9% up). The apparent imbalance is an artifact of high-volume down days contributing disproportionately more posts.

**No sentiment model shows strong separation between up and down days.** Box plots of positive, neutral, and negative scores across all four polarity models reveal substantial overlap between up and down day distributions. FinBERT's neutral score shows the most visible difference.

**Negative sentiment is counterintuitively higher on up days.** Across FinBERT, Twitter-RoBERTa, and Topic-Sentiment, negative scores are slightly higher on up days. This reflects the linguistic culture of r/wallstreetbets, where posts aggressively attacking short sellers register as negative even when the underlying sentiment toward GME is bullish. This is precisely the challenge the reference paper identified.

**Twitter-RoBERTa and Topic-Sentiment are highly redundant.** Pairwise correlations between these models exceed r = 0.75 across all score types, reflecting their shared RoBERTa architecture and social media training domain. VADER captures the most distinct signal of any model despite being the simplest.

**FinBERT is the most consistently correlated model.** FinBERT neutral emerges as the single strongest individual predictor of daily price movement (r = −0.455), appearing five times in the top 20 correlated features. Financial domain adaptation appears to matter more than social media adaptation for this task.

**GoEmotions reveals culturally specific patterns.** Solidarity emotions (love, gratitude) peak on down days, reflecting the "diamond hands" culture, while cognitive processing emotions (realization, surprise, confusion) peak on up days. These dynamics are invisible to positive/negative/neutral models.

**Engagement-weighted sentiment is worse than unweighted.** Weighting by upvotes consistently weakens and sometimes reverses correlations with price movement. Highly upvoted posts tend to be memes and rallying cries, popular but emotionally performative content that does not reflect market conditions.

---

## Results

### Post-Level Baseline

Using the post-level dataset with a proper time-based train/test split (train on the first 80% of trading days, test on the last 20%), all classifiers across all sentiment feature sets produce accuracies between 47% and 51%. This is essentially random guessing. No sentiment model improves over the base features (text metadata and engagement metrics alone). This confirms that once posts from the same day are prevented from appearing in both train and test sets, the predictive signal disappears entirely.

### Daily-Level Classification

Twelve feature sets were evaluated across six classifiers using 3-fold TimeSeriesSplit cross-validation (72 total experiments). The top results:

| Rank | Feature Set | Classifier | CV Accuracy | Std | vs Baseline |
|------|-------------|------------|-------------|-----|-------------|
| 1 | Best + Dispersion | SVM | 0.697 | 0.187 | +0.106 |
| 2 | Base Only | Logistic Regression | 0.667 | 0.113 | +0.076 |
| 3 | VADER | MLP | 0.667 | 0.281 | +0.076 |
| 4 | FinBERT | Logistic Regression | 0.636 | 0.257 | +0.045 |
| 5 | Composite | SVM | 0.636 | 0.223 | +0.045 |

**Base features are competitive with sentiment.** Logistic Regression with just 7 metadata and text structure features (post count, mean upvotes, mean comments, word count, stopword count, average word length, emoji count) achieves 66.7% accuracy with the lowest fold-level standard deviation of any model that beats the baseline. Adding sentiment features does not consistently improve performance.

**No sentiment model clearly dominates.** VADER, FinBERT, RoBERTa, Topic-Sentiment, and GoEmotions all produce similar accuracy ranges. Despite the theoretical advantages of domain-adapted transformers, none translates to meaningfully better price prediction on this dataset.

**More features degrade performance.** PCA-compressed feature sets underperform simpler feature sets. The raw GoEmotions (35 features) and All Models (50 features) sets were excluded entirely due to the dimensionality problem. Feature engineering provided marginal improvement at best.

**Linear models outperform complex ones.** Logistic Regression and SVM are the most consistently strong classifiers. Tree-based models and XGBoost generally underperform, likely because the regularization constraints necessary for n = 44 limit their ability to capture nonlinear patterns.

### Weighted vs. Unweighted

The engagement-weighted dataset produced one high result (Random Forest + VADER at 69.7% in an earlier experiment), but this was not reproducible across feature sets or classifiers. Overall, EDA showed that weighting by upvotes systematically weakens correlations with price movement, and the classification results confirmed that weighted features do not improve prediction.

---

## Discussion

The central finding of this project is that no transformer-based sentiment model meaningfully outperforms VADER for predicting GME's daily stock price direction. This holds across five sentiment models, twelve feature sets, six classifiers, and extensive feature engineering, totaling 72 daily-level experiments plus post-level baselines.

To answer our research question directly: modern transformer-based models do not outperform VADER for this task. Despite domain-specific training on financial text (FinBERT), social media (Twitter-RoBERTa), entity-targeted input (Topic-Sentiment RoBERTa), and Reddit-sourced emotion labels (GoEmotions), no model produces consistently better classification accuracy than the rule-based baseline. In fact, simple metadata features (post count, engagement metrics, text structure) perform as well as any sentiment model, suggesting that community activity levels rather than emotional content carry whatever weak signal exists in this data.

This aligns with what Wang & Luo (2021) observed. Their own results showed that VADER sentiment alone was insufficient for prediction, and their high accuracy depended on semantic embeddings. Our work adds to this by showing that the limitation is not specific to VADER. Even models specifically designed for financial text, social media, or Reddit produce similarly weak predictive signal.

The EDA offers some explanation for why. Post volume trails price movement rather than leading it, suggesting Reddit sentiment is reactive rather than predictive. The linguistic culture of r/wallstreetbets, where profanity and aggression convey positive sentiment and solidarity emotions peak during losses, fundamentally challenges the assumptions of general-purpose sentiment tools, even domain-adapted ones.

A secondary finding concerns the reference paper's methodology. Their stratified train/test split at the post level allowed posts from the same trading day to appear in both sets. When we replicate this setup with a proper time-based split, accuracy drops to chance levels across all models. This suggests that their reported results reflect day-level memorization rather than a learnable sentiment-price relationship.

---

## Limitations

**Small effective sample size.** After aggregating to the daily level, the dataset contains only 44 observations. This severely limits model complexity and makes it difficult to achieve statistically significant results.

**Reverse causality.** Sentiment likely reflects price movement rather than predicting it. People post positive sentiment because the price went up, not the other way around. Without causal inference methods (e.g., Granger causality, instrumental variables), the direction of the relationship cannot be established.

**Narrow time window.** Results are specific to the GME short squeeze period (Jan to Mar 2021), an extraordinary market event. Findings may not generalize to normal market conditions or other stocks.

**Dataset discrepancy.** The Kaggle dataset has been modified since the reference paper was published, with our earliest posts beginning January 28 rather than January 4. While total post counts are nearly identical, the missing early period could affect comparability.

**No embedding experiments.** The reference paper's best result (99.05%) used word2vec embeddings, which we did not replicate. We cannot definitively confirm that the embedding result was inflated by the train/test split methodology, although the same stratified split was used for all of the paper's experiments.

**PCA applied before splitting.** PCA was fit on all 44 rows before cross-validation, which is a minor form of information leakage. Since PCA is unsupervised and does not use the target variable, this is generally accepted but noted for completeness.

---

## Future Work

Several directions could extend this analysis. Training on longer time horizons and multiple stocks would increase sample size and test generalizability. Intraday sentiment analysis, examining sentiment shifts within a trading day rather than daily aggregates, could capture more granular dynamics. Causal inference methods such as Granger causality testing could address the reverse causality concern. Alternative aggregation windows (e.g., hourly, multi-day) could reveal temporal dynamics that daily aggregation misses. Fine-tuning a sentiment model specifically on r/wallstreetbets text, as the reference paper suggested with VADER lexicon customization, could improve sentiment accuracy on this uniquely challenging corpus. Finally, as the paper noted, sentiment may be a better predictor for market indicators other than price direction, such as volume or volatility.

---

## Repository Structure

```
├── README.md
├── notebooks/
│   ├── Feature_Extraction.ipynb         # Notebook 1: Data ingestion, preprocessing, sentiment extraction, aggregation
│   ├── EDA.ipynb                        # Notebook 2: Exploratory data analysis (7 sections)
│   └── Classification_Experiments.ipynb # Notebook 3: Feature engineering, classification, results
├── data/
│   ├── reddit_wsb.csv                             # Raw Reddit data (from Kaggle)
│   ├── df_with_all_sentiments.parquet             # Dataset 1: Post-level with all sentiment scores
│   ├── dataset2_daily_unweighted.parquet          # Dataset 2: Daily aggregated (unweighted)
│   └── dataset3_daily_weighted.parquet            # Dataset 3: Daily aggregated (engagement-weighted)
└── models/
    ├── engineered_results.csv       # Classification results for all 72 experiments
    └── engineered_models.joblib     # Fitted model objects
```

## How to Reproduce

### Requirements
- Python 3.10+
- Google Colab with T4 GPU (required for transformer inference in Notebook 1)
- Key packages: pandas, numpy, scikit-learn, xgboost, transformers, vaderSentiment, yfinance, emoji, matplotlib, seaborn

### Execution Order

1. **Feature_Extraction.ipynb** downloads data, preprocesses text, runs all five sentiment models (approximately 2 to 3 hours on T4 GPU), and produces the three datasets. Checkpoints are saved after each sentiment model in case of runtime disconnection.
2. **EDA.ipynb** loads the aggregated datasets and generates all exploratory visualizations. No GPU required.
3. **Classification_Experiments.ipynb** performs feature engineering, trains and evaluates all classifiers, and produces the results tables. Runs in approximately 5 minutes. GPU is optional (used only for XGBoost acceleration).

---

## References

- Wang, C., & Luo, B. (2021). Predicting $GME Stock Price Movement Using Sentiment from Reddit r/wallstreetbets. *Proceedings of the Third Workshop on Financial Technology and Natural Language Processing (FinNLP@IJCAI 2021)*, 22–30.
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Proceedings of the International AAAI Conference on Web and Social Media*, 8.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models. arXiv:1908.10063.
- Loureiro, D., et al. (2022). TimeLMs: Diachronic Language Models from Twitter. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*.
- Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 4040–4054.
- Antweiler, W. & Frank, M.Z. (2004). Is All That Talk Just Noise? The Information Content of Internet Stock Message Boards. *The Journal of Finance*, 59(3), 1259–1294.
- Anthropic. (2025). Claude (Claude Opus 4) [Large language model]. Used for code development, data analysis, and drafting assistance throughout this project. https://www.anthropic.com
