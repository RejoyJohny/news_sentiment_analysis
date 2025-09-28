
# News Sentiment Analysis Dashboard

This project fetches real-time news headlines using the NewsAPI
, classifies them into positive or negative sentiment using a trained machine learning model, and displays the results on a Streamlit dashboard with filtering options.

* Features

✅ Fetches live news articles from NewsAPI.

✅ Preprocesses headlines & descriptions for classification.

✅ Predicts sentiment using Logistic Regression + TF-IDF pipeline.

✅ Saves processed news with sentiment into CSV (news_stream/).

✅ Interactive Streamlit Dashboard:

** Filter by date range.

** Filter by news source.

** View articles with sentiment labels.

📂 Project Structure/

news_sentiment_analysis/

    │── app.py                # Streamlit dashboard
    │── train_model.py        # Train ML model from dataset
    │── sentiment_dataset.csv # Sample training data
    │── model.pkl             # Trained ML model
    │── news_stream/          # Saved news files (generated at runtime)
    │── processed_news/       # (Optional) Pre-saved processed CSVs
    │── README.md             # Project documentation
