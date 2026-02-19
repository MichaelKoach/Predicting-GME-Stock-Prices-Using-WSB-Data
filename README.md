# Predicting-GME-Stock-Prices-Using-WSB-Data
In this project, I evaluate four RoBERTa-based transformer models against VADER to determine if domain-specific training, media-specific training, topic sentiment, or emotional embedding adds significant predictive power to sentiment analysis. 
This project consists of three phases, corresponding to three notebooks, labeled Feature Engineering, EDA, and Modeling. So far, I have completed the feature engineering and EDA phases.


This study builds on the work of Charlie Wang and Ben Luo in their 2021 paper titled "Predicting $GME Stock Price Movement Using Sentiment from Reddit r/WallStreetBets".
Wang and Luo (2021) used VADER sentiment combined with word2vec and BERT embeddings to predict binary GME price direction (up/down) from 35,726 WSB posts, reporting up to 99% accuracy with a decision tree classifier. 
This project extends their work in three ways:
  1. Multiple sentiment models: in addition to the VADER baseline, I add four models spanning financial-domain, social-media-domain, entity-targeted, and emotion-classification approaches.
  2. Data leakage correction: I addresses a train/test contamination issue in the original methodology where posts from the same trading day appeared in both sets.
  3. Aggregation analysis: I compare post-level classification (replicating the paper) against daily-level aggregation and tests whether engagement-weighted sentiment outperforms equal-weight sentiment.

Data
I joined two datasets for this study. 
  1. A dataset of Reddit posts in r/wallstreetbets collected by Gabriel Preda on Kaggle (https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts?resource=download).
  2. Daily market value data for $GME which I gathered from Yahoo Finance using the yfinance Python library. 

Preprocessing:
My text preprocessing steps follow the original paper: title and body concatenation, emoji replacement with descriptive tags, URL and mention removal, punctuation removal (preserving decimal numbers and emoji tags), lowercasing, and whitespace normalization.


Sentiment Models:
* VADER | General (rule-based) | pos, neu, neg, compound | Paper's original baseline
* FinBERT | Financial text | positive, neutral, negative | Tests whether financial domain adaptation improves prediction on WSB text
* Twitter-RoBERTa | Social media (124M tweets) | positive, neutral, negative | Tests whether social media domain adaptation captures WSB language
* Topic-Sentiment RoBERTa | Social media (entity-targeted) | strongly pos, pos, neutral, neg, strongly neg | Tests whether GME-targeted sentiment outperforms general sentiment
* GoEmotions | Reddit (28 emotion labels) | 28 emotions + neutral | Tests whether granular emotion classification captures dynamics that polarity models miss

Classification Datasets:
I produced 3 datasets for analysis. 
  1. A post-level dataset with 35,735 rows. This mirrors the paper with the addition of the features from the transformer models.
  2. A day-level dataset with 44 rows that aggrigates the average daily values for all featurs.
  3. A day-level dataset with 44 rows that aggrigates the average daily values for all featurs weighted by the number of upvotes.

Key Findings:
I will update this section after classification experiments are complete.

* FinBERT neutral sentiment is the single strongest individual predictor of daily price movement (r = −0.455), outperforming all other sentiment features.
* Engagement-weighted sentiment (weighting by upvotes) consistently weakens correlations with price movement, suggesting that highly upvoted posts are emotionally performative rather than price-informative.
* Twitter-RoBERTa and Topic-Sentiment RoBERTa are highly correlated (r = 0.75–0.89), indicating redundancy; VADER captures the most distinct signal despite being the simplest model.
* GoEmotions reveals culturally specific patterns invisible to polarity models: solidarity emotions (love, gratitude) peak on down days, while cognitive processing emotions (realization, surprise) peak on up days.
* Reddit post volume appears to trail price movement rather than lead it, raising reverse causality concerns about the predictive interpretation of sentiment-price relationships.


References:
Wang, Y., & Luo, T. (2021). Predicting $GME Stock Price Movement Using Sentiment from Reddit r/wallstreetbets.
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models.
Loureiro, D. et al. (2022). TimeLMs: Diachronic Language Models from Twitter.
Demszky, D. et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions.
